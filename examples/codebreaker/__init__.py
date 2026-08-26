"""Codebreaker multi-turn RL example: procedural Mastermind (wordle successor).

Env + scoring are byte-faithful ports from the codebreaker research program
(slime-snapshot examples/codebreaker_multiturn); the harness mirrors
examples/wordle adapted to instance-based play, burn-turn invalid handling,
binary reward, and GRPO with the obo zero-variance rescue.
"""
