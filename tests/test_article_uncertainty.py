def test_article_uncertainty_uses_rounded_bootstrap_half_width():
    from article.source.build import format_uncertainty

    assert format_uncertainty(-0.044807, -0.045968, -0.043711) == '-0.0448 ± 0.0011'
    assert format_uncertainty(0.003578, 0.002022, 0.005323) == '+0.0036 ± 0.0017'
    assert format_uncertainty(0.0, 0.0, 0.0) == '+0 ± 0'
