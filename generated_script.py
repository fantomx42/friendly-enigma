import re

def compile_email_regex():
    """
    Compile a regex pattern for email validation.

    Pattern breakdown:
    - Local part: starts with alphanumeric/allowed chars, dots allowed but not at start/end or consecutive
    - @ separator
    - Domain: alphanumeric with hyphens, dots between labels
    - TLD: 2+ letters
    """
    # Local part: no leading dots, no consecutive dots, no trailing dots
    local_part = r'[a-zA-Z0-9_%+-]+(?:\.[a-zA-Z0-9_%+-]+)*'
    # Domain: labels separated by dots, no leading/trailing hyphens per label
    domain = r'[a-zA-Z0-9](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?'
    domain_with_subdomains = rf'{domain}(?:\.{domain})*'
    # TLD: at least 2 letters
    tld = r'\.[a-zA-Z]{2,}'

    return re.compile(rf'^{local_part}@{domain_with_subdomains}{tld}$')

EMAIL_REGEX = compile_email_regex()

def validate_email(email):
    """
    Validate an email address string.

    Args:
        email: The string to validate

    Returns:
        True if valid email format, False otherwise
    """
    if not isinstance(email, str):
        return False
    if not email or len(email) > 254:  # RFC 5321 max length
        return False
    return bool(EMAIL_REGEX.match(email))


if __name__ == "__main__":
    # Comprehensive test cases
    test_cases = [
        # Valid emails
        ('user@example.com', True),
        ('valid.user+123@example.co.uk', True),
        ('test@subdomain.example.org', True),
        ('firstname.lastname@company.com', True),
        ('email_with_underscore@domain.com', True),
        ('plus+tag@gmail.com', True),
        ('numbers123@test456.com', True),

        # Invalid emails - missing parts
        ('invalid@domain', False),          # Missing TLD
        ('@example.com', False),             # Missing local part
        ('user@', False),                    # Missing domain
        ('userexample.com', False),          # Missing @

        # Invalid emails - bad local part
        ('..test@example.com', False),       # Leading dots
        ('test..user@example.com', False),   # Consecutive dots
        ('test.@example.com', False),        # Trailing dot
        ('.test@example.com', False),        # Leading dot

        # Invalid emails - bad domain
        ('user@.com', False),                # Leading dot in domain
        ('user@-example.com', False),        # Leading hyphen
        ('user@example-.com', False),        # Trailing hyphen

        # Invalid emails - other
        ('', False),                         # Empty string
        ('not an email', False),             # Spaces
        ('user@exam ple.com', False),        # Space in domain
        (None, False),                       # None type
        (123, False),                        # Integer type
    ]

    print("Email Validation Test Results")
    print("=" * 50)

    passed = 0
    failed = 0

    for email, expected in test_cases:
        result = validate_email(email)
        status = "✓" if result == expected else "✗"
        if result == expected:
            passed += 1
        else:
            failed += 1
        print(f"{status} {repr(email)}: {result} (expected {expected})")

    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
