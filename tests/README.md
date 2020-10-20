# Unit tests
Unit tests use the [unittest](https://docs.python.org/3/library/unittest.html) testing framework.

## Test guidelines
* Each unit test should ideally test just one thing.
* Unit tests should be easy to read, even if it requires some code duplication.
* Unit tests should be as simple as possible, so they don't become a burden to support.

[More good unit testing practices](https://pylonsproject.org/community-unit-testing-guidelines.html).

# Unit tests
Run unit tests.
```cd tests & python -m unittest unit_tests```

# Integration tests
Run integration tests.
```cd tests & python -m unittest ci_tests```

# Run all tests
Run all tests.
```cd tests & python -m unittest```

