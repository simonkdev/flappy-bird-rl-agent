from train.training import gamma_for_epoch


def main():
    assert gamma_for_epoch(1, 0.995, 0.9999, 160) == 0.995
    assert gamma_for_epoch(160, 0.995, 0.9999, 160) == 0.9999
    assert gamma_for_epoch(200, 0.995, 0.9999, 160) == 0.9999
    assert gamma_for_epoch(20, 0.995) == 0.995

    gamma_at_81 = gamma_for_epoch(81, 0.995, 0.9999, 160)
    horizon_at_81 = 1.0 / (1.0 - gamma_at_81)
    expected_horizon = 200.0 * ((10_000.0 / 200.0) ** (80.0 / 159.0))
    assert abs(horizon_at_81 - expected_horizon) < 1e-8
    print("gamma schedule tests passed")


if __name__ == "__main__":
    main()
