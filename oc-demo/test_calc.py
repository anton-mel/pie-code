from calc import add, subtract, multiply, divide


def test_all():
    assert add(2, 3) == 5
    assert subtract(5, 3) == 2
    assert multiply(4, 3) == 12
    assert divide(8, 2) == 4


if __name__ == "__main__":
    test_all()
    print("all tests passed")
