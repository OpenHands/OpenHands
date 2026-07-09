"""Contract checks for the OpenHands chart image-bump caller."""

from pathlib import Path

WORKFLOW = Path(__file__).parents[2] / '.github/workflows/bump-chart.yml'


def test_openhands_release_syncs_the_chart_app_version_with_its_image_tag():
    workflow = WORKFLOW.read_text()
    caller_inputs = workflow.split('    with:\n', maxsplit=1)[1]

    assert '      component: openhands\n' in caller_inputs
    assert '      chart_file: charts/openhands/values.yaml\n' in caller_inputs
    assert '      metadata_file: charts/openhands/Chart.yaml\n' in caller_inputs
    assert '      metadata_path: .appVersion\n' in caller_inputs
    assert '      tag: ${{ github.ref_name }}\n' in caller_inputs
