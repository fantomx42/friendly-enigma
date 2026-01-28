def fibonacci(n):
    if n <= 0:
        return "Input should be a positive integer."
    elif n == 1:
        return 0
    elif n == 2:
        return 1

    prev, curr = 0, 1
    for _ in range(2, n):
        temp = curr
        curr += prev
        prev = temp
    return curr

print(f"The 50th Fibonacci number is: {fibonacci(50)}")
