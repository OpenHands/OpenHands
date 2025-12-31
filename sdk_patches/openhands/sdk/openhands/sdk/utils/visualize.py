from rich.text import Text


def display_dict(d) -> Text:
    """Create a Rich Text representation of a dictionary.

    This function is deprecated. Use display_json instead.
    """
    return display_json(d)


def display_json(data) -> Text:
    """Create a Rich Text representation of JSON data.

    Handles dictionaries, lists, strings, numbers, booleans, and None values.
    """
    content = Text()

    if isinstance(data, dict):
        for field_name, field_value in data.items():
            if field_value is None:
                continue  # skip None fields
            content.append(f"\n  {field_name}: ", style="bold")
            if isinstance(field_value, str):
                # Handle multiline strings with proper indentation
                if "\n" in field_value:
                    content.append("\n")
                    for line in field_value.split("\n"):
                        content.append(f"    {line}\n")
                else:
                    content.append(f'"{field_value}"')
            elif isinstance(field_value, (list, dict)):
                content.append(str(field_value))
            else:
                content.append(str(field_value))
    elif isinstance(data, list):
        content.append(f"[List with {len(data)} items]\n")
        for i, item in enumerate(data):
            content.append(f"  [{i}]: ", style="bold")
            if isinstance(item, str):
                content.append(f'"{item}"\n')
            else:
                content.append(f"{item}\n")
    elif isinstance(data, str):
        # Handle multiline strings with proper indentation
        if "\n" in data:
            content.append("String:\n")
            for line in data.split("\n"):
                content.append(f"  {line}\n")
        else:
            content.append(f'"{data}"')
    elif data is None:
        content.append("null")
    else:
        # Handle numbers, booleans, and other JSON primitives
        content.append(str(data))

    return content
