info_packets_sample.csv
- public_packet: same-day public event surprise if usable_from_trade_date==date (EPS surprise; parse fail→0). Strict PIT: only usable_from_trade_date ≤ decision date.
- channel_packet: equal-weight mix of supply-graph neighbors' public_packet and neighbors' ret_lag1 (assembled each day by walking edges).
- msg_sign: sign(public) else sign(channel).
- Macro ALL_MACRO events contribute 0 when unparseable.
