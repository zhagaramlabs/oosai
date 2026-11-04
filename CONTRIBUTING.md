# Contributing to Oosai

Oosai is a research prototype of a Tamil voice agent: speech recognition, a Tamil language model and
speech synthesis in one loop, with the safety rules in code. Issues and pull requests are welcome, in
Tamil or English.

## Ways to help

- **Try it in Tamil and report what goes wrong:** the question, what it heard, what it said, and what
  it should have said.
- **Safety rules** (`agent/rules.py`): cases it should refuse, or hand off to a person (108, a doctor,
  an e-Sevai centre).
- **Speed:** the loop is too slow today. README §12 lists what's next.
- **Native ratings** of the speech samples in `eval/`.

## Before a pull request

1. Set up as in README §9 (Python 3.11 and a `uv` virtual environment).
2. Run what your change touches, and say in the pull request what you measured.
3. Keep the change small and in the style of the code around it. New logic gets a small test.
4. Keep README.md current: the "Updated" date, §12 "Next step", and any decision you change.

## Rules that protect people

- **Voices only with written consent.** Never clone or imitate a real person's voice without that
  person's written consent.
- **No data or weights in git.** Recordings, models and results stay in the gitignored folders.
- **Test sets stay private.** Don't add, train on or publish held-out test questions or answers.
- **Local servers bind to 127.0.0.1,** never 0.0.0.0.

By contributing you agree your work is released under the Apache 2.0 licence ([LICENSE](LICENSE)).
