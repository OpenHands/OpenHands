from binaryornot.check import is_binary


def test_binaryornot_handles_non_ascii_binary_file(tmp_path):
    binary_file = tmp_path / 'paper.pdf'
    binary_file.write_bytes(bytes([0, 159, 255]))

    assert is_binary(str(binary_file)) is True
