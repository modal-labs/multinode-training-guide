from pathlib import Path


def test_hello_file_contents():
    path = Path("/app/hello.txt")
    assert path.exists()
    assert path.read_text().strip() == "Hello, world!"
