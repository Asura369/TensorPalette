import os

import pytest
from PIL import Image

from scripts import check_data


def make_image(path, fmt="JPEG", mode="RGB"):
    color = 120 if mode == "L" else (120, 60, 30)
    img = Image.new(mode, (8, 8), color)
    img.save(path, fmt)


def test_check_data_counts_valid_images(tmp_path):
    make_image(os.path.join(tmp_path, "ok1.jpg"))
    make_image(os.path.join(tmp_path, "ok2.jpg"))
    with open(os.path.join(tmp_path, "broken.jpg"), "w") as f:
        f.write("this is not an image")

    valid, deleted = check_data.check_data(str(tmp_path))
    assert valid == 2
    assert deleted == 1
    assert os.path.exists(os.path.join(tmp_path, "broken.jpg"))


def test_check_data_delete_removes_invalid(tmp_path):
    make_image(os.path.join(tmp_path, "ok.jpg"))
    with open(os.path.join(tmp_path, "broken.jpg"), "w") as f:
        f.write("this is not an image")

    valid, deleted = check_data.check_data(str(tmp_path), delete=True)
    assert valid == 1
    assert deleted == 1
    assert not os.path.exists(os.path.join(tmp_path, "broken.jpg"))


def test_check_data_rejects_non_rgb(tmp_path):
    make_image(os.path.join(tmp_path, "gray.jpg"), mode="L")

    valid, deleted = check_data.check_data(str(tmp_path))
    assert valid == 0
    assert deleted == 1


def test_main_exits_nonzero_below_min_valid(tmp_path, capsys, monkeypatch):
    make_image(os.path.join(tmp_path, "gray.jpg"), mode="L")

    monkeypatch.setattr("sys.argv", ["check_data.py", str(tmp_path), "--min-valid", "1"])
    with pytest.raises(SystemExit) as excinfo:
        check_data.main()
    assert excinfo.value.code == 1
    assert "only 0 valid images" in capsys.readouterr().err
