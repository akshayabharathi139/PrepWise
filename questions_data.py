"""
This file contains the question database for different domains.
Each domain has a list of questions, ideal answers, and keywords for evaluation.
"""

QUESTIONS_DATA = {
    "Python Programming": [
        {
            "question": "Explain the difference between lists and tuples in Python.",
            "ideal_answer": "Lists are mutable ordered collections, while tuples are immutable. Lists use square brackets [] and can be modified after creation. Tuples use parentheses () and cannot be changed after creation. Lists have more built-in methods and generally consume more memory than tuples. Tuples are faster for iteration and are hashable, so they can be used as dictionary keys.",
            "keywords": ["mutable", "immutable", "ordered", "brackets", "parentheses", "memory", "hashable", "dictionary keys", "modify", "iteration"]
        },
        {
            "question": "What are decorators in Python and how do they work?",
            "ideal_answer": "Decorators are a design pattern in Python that allow a user to add new functionality to an existing object without modifying its structure. Decorators are usually called before the definition of a function you want to decorate. They are functions that take another function as an argument, extending the behavior of the latter function without explicitly modifying it. They use the @decorator_name syntax. Decorators make use of closures and higher-order functions concepts.",
            "keywords": ["design pattern", "functionality", "modify", "function argument", "behavior", "syntax", "closures", "higher-order", "wrapper", "extend"]
        },
        {
            "question": "Explain Python's GIL (Global Interpreter Lock) and its implications.",
            "ideal_answer": "The Global Interpreter Lock (GIL) is a mutex that protects access to Python objects, preventing multiple threads from executing Python bytecode at once. This means that in CPython (the standard implementation), threads cannot run Python code truly in parallel on multiple CPU cores. The GIL has implications for CPU-bound multi-threaded programs, which won't see performance improvements from threading. However, I/O-bound programs can still benefit from threading despite the GIL, as it's released during I/O operations. For CPU-bound tasks requiring parallelism, the multiprocessing module is often used instead of threading.",
            "keywords": ["mutex", "threads", "bytecode", "CPython", "parallel", "CPU cores", "CPU-bound", "I/O-bound", "performance", "multiprocessing"]
        },
        {
            "question": "How does memory management work in Python?",
            "ideal_answer": "Python uses automatic memory management with a private heap to store objects and data structures. The memory manager allocates heap space for Python objects, while a built-in garbage collector recycles unused memory for other objects. Python uses reference counting as its primary memory management technique. Each object has a reference count that tracks how many references point to it. When this count reaches zero, the object is deallocated. Python also has a cycle detector that handles circular references. Memory management is mostly transparent to the programmer, though understanding it helps in writing efficient code and avoiding memory leaks.",
            "keywords": ["automatic", "private heap", "garbage collector", "reference counting", "deallocated", "cycle detector", "circular references", "transparent", "efficient", "memory leaks"]
        },
        {
            "question": "What are Python generators and how do they differ from regular functions?",
            "ideal_answer": "Generators are special functions that return an iterator that yields items one at a time, using the 'yield' statement instead of 'return'. Unlike regular functions that compute a value and return it, generators produce a sequence of values over time. They maintain their state between calls, remembering where they left off. This makes generators memory-efficient for working with large datasets or infinite sequences, as they don't compute all values at once but generate them on-demand. Generator expressions are similar to list comprehensions but use parentheses instead of square brackets and create generators instead of lists.",
            "keywords": ["iterator", "yield", "sequence", "state", "memory-efficient", "large datasets", "infinite sequences", "on-demand", "generator expressions", "lazy evaluation"]
        },
        {
            "question": "Explain the concept of duck typing in Python.",
            "ideal_answer": "Duck typing is a programming concept used in Python where the type or class of an object is less important than the methods it defines or the operations it supports. The name comes from the saying, 'If it walks like a duck and quacks like a duck, then it probably is a duck.' In duck typing, we don't check the type of an object; instead, we check whether it has certain methods or properties. This allows for more flexible code that works with any object that supports the required operations, regardless of its actual type. Duck typing is a key aspect of Python's dynamic typing system and supports polymorphism without inheritance.",
            "keywords": ["type", "class", "methods", "operations", "flexible", "dynamic typing", "polymorphism", "inheritance", "interface", "behavior"]
        },
        {
            "question": "What are context managers in Python and how do you implement one?",
            "ideal_answer": "Context managers in Python are objects that define the methods __enter__() and __exit__() to establish a context for a block of code. They're typically used with the 'with' statement to ensure resources are properly managed, like automatically closing files or releasing locks. The __enter__() method is called when entering the context and can return a value that's assigned to the variable after 'as'. The __exit__() method is called when exiting the context (even if an exception occurs) and handles cleanup. You can implement a context manager by creating a class with these methods or by using the contextlib.contextmanager decorator on a generator function that yields once.",
            "keywords": ["__enter__", "__exit__", "with statement", "resources", "cleanup", "exception handling", "contextlib", "decorator", "generator", "yield"]
        },
        {
            "question": "How would you handle exceptions in Python?",
            "ideal_answer": "In Python, exceptions are handled using try-except blocks. The try block contains code that might raise an exception, while the except block catches and handles specific exceptions. You can catch multiple exception types either in separate except blocks or in a single block with a tuple. The else clause executes if no exceptions occur, and the finally clause always executes, regardless of whether an exception occurred. It's best practice to catch specific exceptions rather than using a bare except, which catches all exceptions including KeyboardInterrupt and SystemExit. You can also raise exceptions with the 'raise' statement and create custom exception classes by inheriting from Exception.",
            "keywords": ["try-except", "catch", "multiple exceptions", "else clause", "finally clause", "specific exceptions", "bare except", "raise", "custom exceptions", "inheritance"]
        },
        {
            "question": "Explain Python's approach to object-oriented programming.",
            "ideal_answer": "Python is a multi-paradigm language that supports object-oriented programming (OOP). In Python, everything is an object, including functions and classes themselves. Python's OOP approach includes class-based inheritance, encapsulation, polymorphism, and to some extent, abstraction. Classes are defined with the 'class' keyword, and instances are created by calling the class. Python supports single and multiple inheritance, with the Method Resolution Order (MRO) determining which method to call in complex inheritance hierarchies. Special methods (dunder methods) like __init__, __str__, and __eq__ customize object behavior. Python uses name mangling (with double underscores) for private attributes, though true private encapsulation is not enforced. Properties, class methods, and static methods provide additional OOP functionality.",
            "keywords": ["class", "inheritance", "encapsulation", "polymorphism", "abstraction", "instances", "multiple inheritance", "MRO", "dunder methods", "properties"]
        },
        {
            "question": "What are some best practices for writing efficient Python code?",
            "ideal_answer": "For efficient Python code: Use appropriate data structures (dictionaries for lookups, sets for membership tests). Leverage built-in functions and libraries which are optimized in C. Use list/dictionary comprehensions instead of loops when appropriate. Avoid global variables and excessive function calls in loops. Use generators for large datasets to save memory. Profile your code to identify bottlenecks using tools like cProfile. Consider using NumPy for numerical operations. Minimize I/O operations and use buffering. Use string joining instead of concatenation in loops. For CPU-bound tasks, consider multiprocessing, Cython, or PyPy. Follow the principle 'premature optimization is the root  consider multiprocessing, Cython, or PyPy. Follow the principle 'premature optimization is the root of all evil' - write clear code first, then optimize if necessary.",
            "keywords": ["data structures", "built-in functions", "comprehensions", "global variables", "generators", "profiling", "NumPy", "I/O operations", "string joining", "multiprocessing"]
        }
    ],
    "Data Science": [
        # Data Science questions would be listed here
    ],
    "Web Development": [
        # Web Development questions would be listed here
    ],
    "Machine Learning": [
        # Machine Learning questions would be listed here
    ]
}