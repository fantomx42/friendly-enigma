import re

def compile_email_regex():
    return re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

EMAIL_REGEX = compile_email_regex()

def validate_email(email):
    if not isinstance(email, str):
        return False
    return bool(EMAIL_REGEX.match(email))

# Example usage:
if __name__ == "__main__":
    test_emails = [
        'user@example.com',
        'valid.user+123@example.co.uk',
        'invalid@domain',  # Missing TLD
        'another@.com',  # Invalid domain start
        '..test@example.com'  # Invalid local part start
    ]

    for email in test_emails:
        print(f"{email}: {'Valid' if validate_email(email) else 'Invalid'}")
