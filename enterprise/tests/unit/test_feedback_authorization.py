"""
Test for CWE-862: Missing authorization on feedback submission endpoint.

The submit_conversation_feedback endpoint must verify that the authenticated
user owns the conversation before allowing feedback submission.

Uses AST-based analysis to work around complex enterprise dependency chains.
"""
import ast
import os

import pytest


def _get_feedback_source():
    """Read the feedback route source file."""
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(base, 'server', 'routes', 'feedback.py')
    with open(path) as f:
        return f.read()


def test_submit_conversation_feedback_has_user_id_parameter():
    """
    Verify submit_conversation_feedback declares a user_id parameter.

    Before the fix the signature was:
        async def submit_conversation_feedback(feedback: FeedbackRequest)
    After the fix:
        async def submit_conversation_feedback(feedback: FeedbackRequest, user_id: str = Depends(get_user_id))
    """
    source = _get_feedback_source()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == 'submit_conversation_feedback':
                param_names = [arg.arg for arg in node.args.args]
                assert 'user_id' in param_names, (
                    'submit_conversation_feedback must accept a user_id parameter '
                    'for authorization checks (CWE-862)'
                )
                return

    pytest.fail('submit_conversation_feedback function not found in source')


def test_submit_conversation_feedback_performs_ownership_check():
    """
    Verify that submit_conversation_feedback calls an ownership verification
    function that queries StoredConversationMetadataSaas with user_id.

    The check may be inline or delegated to a helper — either way the
    function must use user_id for an authorization-related call.
    """
    source = _get_feedback_source()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == 'submit_conversation_feedback':
                body_source = ast.get_source_segment(source, node)
                assert body_source is not None, 'Could not extract function source'

                # The function must reference user_id in some ownership-checking call.
                # It can either:
                #   (a) inline the StoredConversationMetadataSaas query, or
                #   (b) call a helper like _verify_conversation_ownership
                has_inline_check = 'StoredConversationMetadataSaas' in body_source
                has_helper_call = '_verify_conversation_ownership' in body_source

                assert has_inline_check or has_helper_call, (
                    'submit_conversation_feedback must verify conversation ownership '
                    'either inline or via a helper function'
                )

                # And the helper (if used) must actually exist and do the right thing
                if has_helper_call and not has_inline_check:
                    # Find the helper function and verify it queries with user_id
                    helper_found = False
                    for other_node in ast.walk(tree):
                        if isinstance(
                            other_node, (ast.FunctionDef, ast.AsyncFunctionDef)
                        ) and other_node.name == '_verify_conversation_ownership':
                            helper_source = ast.get_source_segment(source, other_node)
                            assert helper_source is not None
                            assert 'StoredConversationMetadataSaas' in helper_source, (
                                '_verify_conversation_ownership must query '
                                'StoredConversationMetadataSaas'
                            )
                            assert 'user_id' in helper_source, (
                                '_verify_conversation_ownership must filter by user_id'
                            )
                            helper_found = True
                            break
                    assert helper_found, (
                        '_verify_conversation_ownership helper function not found'
                    )
                return

    pytest.fail('submit_conversation_feedback function not found in source')


def test_ownership_check_raises_404_on_missing_conversation():
    """
    Verify that the ownership verification raises HTTP 404 when the
    conversation is not found for the authenticated user.
    """
    source = _get_feedback_source()
    tree = ast.parse(source)

    # Check the helper or inline — wherever the 404 is raised
    ownership_functions = ['submit_conversation_feedback', '_verify_conversation_ownership']
    found_404 = False

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in ownership_functions:
                body_source = ast.get_source_segment(source, node)
                if body_source and (
                    'HTTP_404_NOT_FOUND' in body_source or '404' in body_source
                ):
                    found_404 = True
                    break

    assert found_404, (
        'The ownership verification path must raise HTTP 404 when the '
        'conversation is not found for the authenticated user'
    )
