import pytest
import numpy as np

from keinsum import parse, construct, keinsum


def test_parse():
    assert parse("Ik,kJ", np.ones((2, 3, 4)), np.ones((4, 5, 6))) == [
        [0, 1, 4],
        [4, 2, 3],
    ]
    assert parse("Ik,Ij", np.ones((2, 3, 4)), np.ones((2, 3, 5))) == [
        [0, 1, 3],
        [0, 1, 2],
    ]
    assert parse("Ijk,jkL", np.ones((2, 3, 4, 5)), np.ones((4, 5, 6, 7))) == [
        [0, 1, 2, 3],
        [2, 3, 4, 5],
    ]
    assert parse("iJ,iK->JK", np.ones((2, 3, 4)), np.ones((2, 5, 6))) == [
        [0, 1, 2],
        [0, 3, 4],
        [1, 2, 3, 4],
    ]
    assert parse("iJ,iK->KJ", np.ones((2, 3, 4)), np.ones((2, 5, 6))) == [
        [0, 1, 2],
        [0, 3, 4],
        [3, 4, 1, 2],
    ]

    with pytest.raises(ValueError):
        parse("i->j")


def test_keinsum():
    a, b = np.random.randn(2, 3, 4), np.random.randn(4, 5, 6)
    assert np.allclose(keinsum("Ik,kJ", a, b), np.einsum("ijk,klm", a, b))

    a, b = np.random.randn(2, 3, 4), np.random.randn(2, 3, 5)
    assert np.allclose(keinsum("Ik,Ij", a, b), np.einsum("lmk,lmj", a, b))

    a, b = np.ones((2, 3, 4, 5)), np.ones((4, 5, 6, 7))
    assert np.allclose(keinsum("Ijk,jkL", a, b), np.einsum("abjk,jkcd", a, b))

    a, b = np.ones((2, 3, 4)), np.ones((2, 5, 6))
    assert np.allclose(keinsum("iJ,iK->JK", a, b), np.einsum("ijk,ilm->jklm", a, b))
    assert np.allclose(keinsum("iJ,iK->KJ", a, b), np.einsum("ijk,ilm->lmjk", a, b))

    with pytest.raises(ValueError):
        parse('iJ,Jkl', a, b)


if __name__ == "__main__":
    pytest.main(["-s", __file__])  # + '::test7'])
