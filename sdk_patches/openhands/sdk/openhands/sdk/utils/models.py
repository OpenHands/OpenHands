import inspect
import json
import logging
import os
from abc import ABC
from typing import Annotated, Any, ClassVar, Literal, NoReturn, Self, Union

from pydantic import (
    BaseModel,
    Discriminator,
    Field,
    Tag,
    TypeAdapter,
    ValidationError,
)
from pydantic_core import ErrorDetails


logger = logging.getLogger(__name__)
_rebuild_required = True


def _is_abstract(type_: type) -> bool:
    """Determine whether the class directly extends ABC or contains abstract methods"""
    try:
        return inspect.isabstract(type_) or ABC in type_.__bases__
    except Exception:
        return False


def _get_all_subclasses(cls) -> set[type]:
    """
    Recursively finds and returns all (loaded) subclasses of a given class.
    """
    result = set()
    for subclass in cls.__subclasses__():
        result.add(subclass)
        result.update(_get_all_subclasses(subclass))
    return result


def rebuild_all():
    """Rebuild all polymorphic classes."""
    global _rebuild_required
    _rebuild_required = False
    for cls in _get_all_subclasses(OpenHandsModel):
        cls.model_rebuild(force=True)
    for cls in _get_all_subclasses(DiscriminatedUnionMixin):
        cls.model_rebuild(force=True)


def kind_of(obj) -> str:
    """Get the string value for the kind tag"""
    if isinstance(obj, dict):
        return obj["kind"]
    if not hasattr(obj, "__name__"):
        obj = obj.__class__
    return obj.__name__


def _create_enhanced_discriminated_union_error_message(
    invalid_kind: str, cls_name: str, valid_kinds: list[str]
) -> str:
    """Create an enhanced error message for discriminated union validation failures."""
    possible_kinds_str = ", ".join(sorted(valid_kinds)) if valid_kinds else "none"
    return (
        f"Unexpected kind '{invalid_kind}' for {cls_name}. "
        f"Expected one of: {possible_kinds_str}. "
        f"If you receive this error when trying to wrap a "
        f"DiscriminatedUnion instance inside another pydantic model, "
        f"you may need to use OpenHandsModel instead of BaseModel "
        f"to make sure that an invalid schema has not been cached."
    )


def _extract_invalid_kind_from_validation_error(error: ErrorDetails) -> str:
    """Extract the invalid kind from a Pydantic validation error."""
    input_value = error.get("input")
    if input_value is not None and hasattr(input_value, "kind"):
        return input_value.kind
    elif isinstance(input_value, dict) and "kind" in input_value:
        return input_value["kind"]
    else:
        return kind_of(input_value)


def _handle_discriminated_union_validation_error(
    validation_error: ValidationError, cls_name: str, valid_kinds: list[str]
) -> NoReturn:
    """Handle discriminated union validation errors with enhanced messages."""
    for error in validation_error.errors():
        if error.get("type") == "union_tag_invalid":
            invalid_kind = _extract_invalid_kind_from_validation_error(error)
            error_msg = _create_enhanced_discriminated_union_error_message(
                invalid_kind, cls_name, valid_kinds
            )
            raise ValueError(error_msg) from validation_error

    # If it's not a discriminated union error, re-raise the original error
    raise validation_error


def get_known_concrete_subclasses(cls) -> list[type]:
    """Recursively returns all concrete subclasses in a stable order,
    without deduping classes that share the same (module, name)."""
    out: list[type] = []
    for sub in cls.__subclasses__():
        # Recurse first so deeper classes appear after their parents
        out.extend(get_known_concrete_subclasses(sub))
        if not _is_abstract(sub):
            out.append(sub)

    # Use qualname to distinguish nested/local classes (like test-local Cat)
    out.sort(key=lambda t: (t.__module__, getattr(t, "__qualname__", t.__name__)))
    return out


class OpenHandsModel(BaseModel):
    """
    Tags a class where the which may be a discriminated union or contain fields
    which contain a discriminated union. The first time an instance is initialized,
    the schema is loaded, or a model is validated after a subclass is defined we
    regenerate all the polymorphic mappings.
    """

    def model_post_init(self, _context):
        _rebuild_if_required()

    @classmethod
    def model_validate(cls, *args, **kwargs) -> Self:
        _rebuild_if_required()
        return super().model_validate(*args, **kwargs)

    @classmethod
    def model_validate_json(cls, *args, **kwargs) -> Self:
        _rebuild_if_required()
        return super().model_validate_json(*args, **kwargs)

    @classmethod
    def model_json_schema(cls, *args, **kwargs) -> dict[str, Any]:
        _rebuild_if_required()
        return super().model_json_schema(*args, **kwargs)

    def model_dump_json(self, **kwargs):
        # This was overridden because it seems there is a bug where sometimes
        # duplicate fields are produced by model_dump_json which does not appear
        # in model_dump
        kwargs["mode"] = "json"
        return json.dumps(self.model_dump(**kwargs), ensure_ascii=False)

    def __init_subclass__(cls, **kwargs):
        """
        When a new subclass is defined, mark that we will need
        to rebuild everything
        """
        global _rebuild_required
        _rebuild_required = True

        return super().__init_subclass__(**kwargs)


class DiscriminatedUnionMixin(OpenHandsModel, ABC):
    """A Base class for members of tagged unions discriminated by the class name.

    This class provides automatic subclass registration and discriminated union
    functionality. Each subclass is automatically registered when defined and
    can be used for polymorphic serialization/deserialization.

    Child classes will automatically have a type field defined, which is used as a
    discriminator for union types.
    """

    __pydantic_core_schema__: ClassVar[Any]
    __pydantic_validator__: ClassVar[Any]
    __pydantic_serializer__: ClassVar[Any]

    kind: str = Field(default="")  # We dynamically update on a per class basis

    @classmethod
    def resolve_kind(cls, kind: str) -> type:
        for subclass in get_known_concrete_subclasses(cls):
            if subclass.__name__ == kind:
                return subclass

        # Generate enhanced error message for unknown kind
        valid_kinds = [
            subclass.__name__ for subclass in get_known_concrete_subclasses(cls)
        ]
        error_msg = _create_enhanced_discriminated_union_error_message(
            kind, cls.__name__, valid_kinds
        )
        raise ValueError(error_msg)

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        """Generate discriminated union schema for TypeAdapter compatibility."""
        if cls.__name__ == "DiscriminatedUnionMixin":
            return handler(source_type)

        if _is_abstract(source_type):
            _rebuild_if_required()
            serializable_type = source_type.get_serializable_type()
            # If there are subclasses, generate schema for the discriminated union
            if serializable_type is not source_type:
                from pydantic_core import core_schema

                # Generate the base schema
                base_schema = handler.generate_schema(serializable_type)

                # Wrap it with a custom validation function that provides
                # enhanced error messages
                def validate_with_enhanced_error(value, handler_func, info):  # noqa: ARG001
                    try:
                        return handler_func(value)
                    except ValidationError as e:
                        valid_kinds = [
                            subclass.__name__
                            for subclass in get_known_concrete_subclasses(source_type)
                        ]
                        _handle_discriminated_union_validation_error(
                            e, source_type.__name__, valid_kinds
                        )

                # Create a with_info_wrap_validator_function schema
                return core_schema.with_info_wrap_validator_function(
                    validate_with_enhanced_error,
                    base_schema,
                )

        return handler(source_type)

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        """Add discriminator to OpenAPI schema and ensure component generation."""
        json_schema = handler(core_schema)

        # Add discriminator if this is a oneOf schema
        if isinstance(json_schema, dict) and "oneOf" in json_schema:
            # Add title for abstract classes to encourage separate component creation
            if _is_abstract(cls) and "title" not in json_schema:
                json_schema["title"] = cls.__name__

            if "discriminator" not in json_schema:
                mapping = {}
                for option in json_schema["oneOf"]:
                    if "$ref" in option:
                        kind = option["$ref"].split("/")[-1]
                        mapping[kind] = option["$ref"]

                if mapping:
                    json_schema["discriminator"] = {
                        "propertyName": "kind",
                        "mapping": mapping,
                    }

        return json_schema

    @classmethod
    def model_rebuild(
        cls,
        *,
        force=False,
        raise_errors=True,
        _parent_namespace_depth=2,
        _types_namespace=None,
    ):
        if cls == DiscriminatedUnionMixin:
            pass
        if _is_abstract(cls):
            subclasses = get_known_concrete_subclasses(cls)
            kinds = [subclass.__name__ for subclass in subclasses]
            if kinds:
                kind_field = cls.model_fields["kind"]
                kind_field.annotation = Literal[tuple(kinds)]  # type: ignore
                kind_field.default = kinds[0]

            type_adapter = TypeAdapter(cls.get_serializable_type())
            cls.__pydantic_core_schema__ = type_adapter.core_schema
            cls.__pydantic_validator__ = type_adapter.validator
            cls.__pydantic_serializer__ = type_adapter.serializer
            return

        return super().model_rebuild(
            force=force,
            raise_errors=raise_errors,
            _parent_namespace_depth=_parent_namespace_depth,
            _types_namespace=_types_namespace,
        )

    @classmethod
    def get_serializable_type(cls) -> type:
        """
        Custom method to get the union of all currently loaded
        non absract subclasses
        """

        # If the class is not abstract return self
        if not _is_abstract(cls):
            return cls

        subclasses = list(get_known_concrete_subclasses(cls))
        if not subclasses:
            return cls

        if len(subclasses) == 1:
            # Returning the concrete type ensures Pydantic instantiates the subclass
            # (e.g. Agent) rather than the abstract base (e.g. AgentBase) when there is
            # only ONE concrete subclass.
            return subclasses[0]

        serializable_type = Annotated[
            Union[*tuple(Annotated[t, Tag(t.__name__)] for t in subclasses)],
            Discriminator(kind_of),
        ]
        return serializable_type  # type: ignore

    @classmethod
    def model_validate(cls, obj: Any, **kwargs) -> Self:
        try:
            if _is_abstract(cls):
                resolved = cls.resolve_kind(kind_of(obj))
            else:
                resolved = super()
            result = resolved.model_validate(obj, **kwargs)
            return result  # type: ignore
        except ValidationError as e:
            valid_kinds = [
                subclass.__name__ for subclass in get_known_concrete_subclasses(cls)
            ]
            _handle_discriminated_union_validation_error(e, cls.__name__, valid_kinds)

    @classmethod
    def model_validate_json(
        cls,
        json_data: str | bytes | bytearray,
        **kwargs,
    ) -> Self:
        data = json.loads(json_data)
        if _is_abstract(cls):
            resolved = cls.resolve_kind(kind_of(data))
        else:
            resolved = super()
        result = resolved.model_validate(data, **kwargs)
        return result  # type: ignore

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # If concrete, stamp kind Literal and collision check
        if not _is_abstract(cls):
            # 1) Stamp discriminator
            cls.kind = cls.__name__
            cls.__annotations__["kind"] = Literal[cls.__name__]

            # 2) Collision check
            mro = cls.mro()
            union_class = mro[mro.index(DiscriminatedUnionMixin) - 1]
            concretes = get_known_concrete_subclasses(union_class)  # sorted list
            kinds: dict[str, type] = {}
            for sub in concretes:
                k = kind_of(sub)
                if k in kinds and kinds[k] is not sub:
                    raise ValueError(
                        f"Duplicate kind detected for {union_class} : {cls}, {sub}"
                    )
                kinds[k] = sub

        # Rebuild any abstract union owners in the MRO that rely on subclass sets
        for base in cls.mro():
            # Stop when we pass ourselves
            if base is cls:
                continue
            # Only rebuild abstract DiscriminatedUnion owners
            if (
                isinstance(base, type)
                and issubclass(base, DiscriminatedUnionMixin)
                and _is_abstract(base)
            ):
                base.model_rebuild(force=True)


def _rebuild_if_required():
    if _rebuild_required:
        rebuild_all()


def _extract_discriminated_unions(schema: dict) -> dict:
    """Extract inline discriminated unions as separate components.

    Recursively scans the schema and extracts any inline discriminated union
    (oneOf + discriminator + title) as a separate component, replacing it with a $ref.
    Also deduplicates schemas with identical titles.
    """
    import json
    import re
    from collections import defaultdict

    if not isinstance(schema, dict):
        return schema

    # OpenAPI schema names must match this pattern
    valid_name_pattern = re.compile(r"^[a-zA-Z0-9._-]+$")

    schemas = schema.get("components", {}).get("schemas", {})
    extracted = {}

    def _find_and_extract(obj, path=""):
        if not isinstance(obj, dict):
            return obj

        # Extract inline discriminated unions
        if "oneOf" in obj and "discriminator" in obj and "title" in obj:
            title = obj["title"]
            if (
                title not in schemas
                and title not in extracted
                and valid_name_pattern.match(title)
            ):
                extracted[title] = {
                    "oneOf": obj["oneOf"],
                    "discriminator": obj["discriminator"],
                    "title": title,
                }
                return {"$ref": f"#/components/schemas/{title}"}

        # Recursively process nested structures
        result = {}
        for key, value in obj.items():
            if isinstance(value, dict):
                result[key] = _find_and_extract(value, f"{path}.{key}")
            elif isinstance(value, list):
                result[key] = [
                    _find_and_extract(item, f"{path}.{key}[]") for item in value
                ]
            else:
                result[key] = value
        return result

    schema = _find_and_extract(schema)

    if extracted and "components" in schema and "schemas" in schema["components"]:
        schema["components"]["schemas"].update(extracted)

    # Deduplicate schemas with same title (prefer *-Output over *-Input over base)
    schemas = schema.get("components", {}).get("schemas", {})
    title_to_names = defaultdict(list)
    for name, defn in schemas.items():
        if isinstance(defn, dict):
            title_to_names[defn.get("title", name)].append(name)

    to_remove = {}
    for title, names in title_to_names.items():
        if len(names) > 1:
            # Prefer: *-Output > *-Input > base name
            keep = sorted(
                names,
                key=lambda n: (
                    0 if n.endswith("-Output") else 1 if n.endswith("-Input") else 2,
                    n,
                ),
            )[0]
            for name in names:
                if name != keep:
                    to_remove[name] = keep

    if to_remove:
        schema_str = json.dumps(schema)
        for old, new in to_remove.items():
            schema_str = schema_str.replace(
                f'"#/components/schemas/{old}"', f'"#/components/schemas/{new}"'
            )
        schema = json.loads(schema_str)
        for old in to_remove:
            schema["components"]["schemas"].pop(old, None)

    return schema


def _patch_fastapi_discriminated_union_support():
    """Patch FastAPI to handle discriminated union schemas without $ref.

    This ensures discriminated unions from DiscriminatedUnionMixin work correctly
    with FastAPI's OpenAPI schema generation. The patch prevents KeyError when
    FastAPI encounters schemas without $ref keys (which discriminated unions use).

    Also extracts inline discriminated unions as separate schema components for
    better OpenAPI documentation and Swagger UI display.

    Skips patching if SKIP_FASTAPI_DISCRIMINATED_UNION_FIX environment variable is set.
    """
    # Skip patching if environment variable flag is defined
    if os.environ.get("SKIP_FASTAPI_DISCRIMINATED_UNION_FIX"):
        logger.debug(
            "Skipping FastAPI discriminated union patch due to environment variable"
        )
        return

    try:
        import fastapi._compat.v2 as fastapi_v2
        from fastapi import FastAPI

        _original_remap = fastapi_v2._remap_definitions_and_field_mappings

        def _patched_remap_definitions_and_field_mappings(**kwargs):
            """Patched version that handles schemas w/o $ref (discriminated unions)."""
            field_mapping = kwargs.get("field_mapping", {})
            model_name_map = kwargs.get("model_name_map", {})

            # Build old_name -> new_name map, skipping schemas without $ref
            old_name_to_new_name_map = {}
            for field_key, schema in field_mapping.items():
                model = field_key[0].type_
                if model not in model_name_map:
                    continue
                new_name = model_name_map[model]

                # Skip schemas without $ref (discriminated unions)
                if "$ref" not in schema:
                    continue

                old_name = schema["$ref"].split("/")[-1]
                if old_name in {f"{new_name}-Input", f"{new_name}-Output"}:
                    continue
                old_name_to_new_name_map[old_name] = new_name

            # Replace refs using FastAPI's helper
            from fastapi._compat.v2 import _replace_refs

            new_field_mapping = {}
            for field_key, schema in field_mapping.items():
                new_schema = _replace_refs(
                    schema=schema,
                    old_name_to_new_name_map=old_name_to_new_name_map,
                )
                new_field_mapping[field_key] = new_schema

            definitions = kwargs.get("definitions", {})
            new_definitions = {}
            for key, value in definitions.items():
                new_key = old_name_to_new_name_map.get(key, key)
                new_value = _replace_refs(
                    schema=value,
                    old_name_to_new_name_map=old_name_to_new_name_map,
                )
                new_definitions[new_key] = new_value

            return new_field_mapping, new_definitions

        # Apply the patch
        fastapi_v2._remap_definitions_and_field_mappings = (
            _patched_remap_definitions_and_field_mappings
        )

        # Patch FastAPI.openapi() to extract discriminated unions
        _original_openapi = FastAPI.openapi

        def _patched_openapi(self):
            """Patched openapi() that extracts discriminated unions."""
            schema = _original_openapi(self)
            return _extract_discriminated_unions(schema)

        FastAPI.openapi = _patched_openapi

    except (ImportError, AttributeError):
        # FastAPI not available or internal API changed
        pass


# Always call the FastAPI patch after DiscriminatedUnionMixin definition
_patch_fastapi_discriminated_union_support()
