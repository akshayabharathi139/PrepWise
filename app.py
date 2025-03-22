import streamlit as st
import pandas as pd
import numpy as np
import json
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import random
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
nltk.download('punkt_tab')
import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')



# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Set page configuration
st.set_page_config(
    page_title="Mock Interview Assistant",
    page_icon="🎯",
    layout="wide"
)

# Initialize session state variables
if 'current_question' not in st.session_state:
    st.session_state.current_question = 0
if 'answers' not in st.session_state:
    st.session_state.answers = []
if 'scores' not in st.session_state:
    st.session_state.scores = []
if 'interview_complete' not in st.session_state:
    st.session_state.interview_complete = False
if 'domain' not in st.session_state:
    st.session_state.domain = None
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'question_times' not in st.session_state:
    st.session_state.question_times = []

# Load question data
@st.cache_data
def load_questions():
    # In a real application, this would load from a database or JSON file
    questions = {
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
                "ideal_answer": "For efficient Python code: Use appropriate data structures (dictionaries for lookups, sets for membership tests). Leverage built-in functions and libraries which are optimized in C. Use list/dictionary comprehensions instead of loops when appropriate. Avoid global variables and excessive function calls in loops. Use generators for large datasets to save memory. Profile your code to identify bottlenecks using tools like cProfile. Consider using NumPy for numerical operations. Minimize I/O operations and use buffering. Use string joining instead of concatenation in loops. For CPU-bound tasks, consider multiprocessing, Cython, or PyPy. Follow the principle 'premature optimization is the root of all evil' - write clear code first, then optimize if necessary.",
                "keywords": ["data structures", "built-in functions", "comprehensions", "global variables", "generators", "profiling", "NumPy", "I/O operations", "string joining", "multiprocessing"]
            }
        ],
        "Data Science": [
            {
                "question": "Explain the difference between supervised and unsupervised learning.",
                "ideal_answer": "Supervised learning uses labeled data where the algorithm learns to map inputs to known outputs, like classification or regression problems. The algorithm is 'supervised' because it's trained on data with known correct answers. Examples include linear regression, decision trees, and neural networks for prediction tasks. Unsupervised learning works with unlabeled data to find patterns or structures without predefined outputs. The algorithm must discover relationships on its own. Common unsupervised techniques include clustering (like K-means), dimensionality reduction (like PCA), and association rule learning. Semi-supervised learning combines both approaches, using a small amount of labeled data with a larger amount of unlabeled data.",
                "keywords": ["labeled", "unlabeled", "classification", "regression", "patterns", "clustering", "K-means", "dimensionality reduction", "PCA", "semi-supervised"]
            },
            {
                "question": "What is the bias-variance tradeoff in machine learning?",
                "ideal_answer": "The bias-variance tradeoff is a fundamental concept in machine learning that describes the relationship between a model's ability to fit the training data (bias) and its sensitivity to fluctuations in the training data (variance). High bias models are too simple and underfit the data, missing important patterns and performing poorly on both training and test data. High variance models are too complex and overfit the training data, capturing noise rather than underlying patterns, performing well on training data but poorly on test data. The goal is to find the optimal balance that minimizes total error. Techniques to manage this tradeoff include regularization, cross-validation, ensemble methods, and selecting appropriate model complexity. The tradeoff is visualized by the U-shaped test error curve when plotting error against model complexity.",
                "keywords": ["underfit", "overfit", "complexity", "error", "regularization", "cross-validation", "ensemble", "noise", "patterns", "U-shaped curve"]
            },
            {
                "question": "How would you handle missing data in a dataset?",
                "ideal_answer": "Handling missing data involves several approaches: First, understand the missingness mechanism (MCAR, MAR, or MNAR) and the extent of missing values. For analysis, you can remove rows with missing values (listwise deletion) if data is MCAR and the proportion is small. Alternatively, you can impute missing values using statistical methods like mean/median/mode imputation, or more sophisticated approaches like KNN, regression, or multiple imputation. For categorical data, you might create a new 'Missing' category. Another approach is to use algorithms that handle missing values natively, like certain decision tree implementations. You can also add a binary indicator variable to flag where values were missing. The best approach depends on the data, the analysis goals, and the missingness pattern.",
                "keywords": ["MCAR", "MAR", "MNAR", "listwise deletion", "imputation", "mean", "median", "KNN", "multiple imputation", "indicator variable"]
            },
            {
                "question": "Explain the concept of regularization in machine learning models.",
                "ideal_answer": "Regularization is a technique used to prevent overfitting in machine learning models by adding a penalty term to the loss function that discourages complex models. The two most common forms are L1 (Lasso) and L2 (Ridge) regularization. L1 adds the sum of absolute values of coefficients, promoting sparsity by driving some coefficients to exactly zero, effectively performing feature selection. L2 adds the sum of squared coefficients, shrinking all coefficients toward zero but rarely to exactly zero. The regularization strength is controlled by a hyperparameter (often denoted as alpha or lambda) that balances the original loss function with the penalty term. Higher values increase regularization. Other forms include Elastic Net (combining L1 and L2), dropout in neural networks, and early stopping. Regularization is especially important for high-dimensional data or complex models.",
                "keywords": ["overfitting", "penalty", "L1", "L2", "Lasso", "Ridge", "sparsity", "feature selection", "hyperparameter", "Elastic Net"]
            },
            {
                "question": "What is the curse of dimensionality and how does it affect machine learning?",
                "ideal_answer": "The curse of dimensionality refers to various phenomena that arise when analyzing data in high-dimensional spaces that do not occur in low-dimensional settings. As dimensions increase, the volume of the space increases exponentially, making data increasingly sparse. This sparsity is problematic for methods that require statistical significance. Distance metrics become less meaningful as most points become equidistant from each other. Models require exponentially more data to maintain the same level of accuracy. Feature interactions grow combinatorially. To address these issues, we use dimensionality reduction techniques like PCA or t-SNE, feature selection methods, regularization, or algorithms less affected by high dimensions like tree-based methods. The curse of dimensionality highlights why feature engineering and selection are crucial in machine learning.",
                "keywords": ["high-dimensional", "sparsity", "distance metrics", "data requirements", "feature interactions", "PCA", "t-SNE", "feature selection", "regularization", "tree-based methods"]
            },
            {
                "question": "Describe the process of cross-validation and why it's important.",
                "ideal_answer": "Cross-validation is a resampling technique used to evaluate machine learning models on limited data. The most common form is k-fold cross-validation, where the dataset is divided into k equal folds. The model is trained on k-1 folds and validated on the remaining fold, repeating this process k times with each fold serving as the validation set once. The results are averaged to provide a more robust performance estimate. Cross-validation is important because it helps detect overfitting, provides a more reliable estimate of model performance on unseen data than a single train-test split, reduces bias and variance in model evaluation, and helps in hyperparameter tuning. Variations include stratified k-fold (preserves class distribution), leave-one-out (k equals sample size), and time series cross-validation for temporal data.",
                "keywords": ["k-fold", "resampling", "overfitting", "performance estimate", "bias", "variance", "hyperparameter tuning", "stratified", "leave-one-out", "time series"]
            },
            {
                "question": "How would you evaluate a classification model's performance?",
                "ideal_answer": "Evaluating a classification model involves multiple metrics, as no single metric captures all aspects of performance. For balanced datasets, accuracy (proportion of correct predictions) is useful. For imbalanced data, precision (positive predictive value), recall (sensitivity), and F1-score (harmonic mean of precision and recall) are more informative. The confusion matrix shows true/false positives/negatives. ROC curves plot true positive rate against false positive rate at various thresholds, with AUC measuring overall performance. Precision-Recall curves are better for imbalanced data. Log loss measures the uncertainty of predictions. For multi-class problems, use macro/micro/weighted averaging of metrics. Beyond metrics, consider the business context, cost of different error types, and model interpretability. Cross-validation ensures reliable evaluation, and comparing to baseline models provides context.",
                "keywords": ["accuracy", "precision", "recall", "F1-score", "confusion matrix", "ROC", "AUC", "imbalanced data", "log loss", "baseline"]
            },
            {
                "question": "Explain the concept of feature engineering and why it's important.",
                "ideal_answer": "Feature engineering is the process of transforming raw data into features that better represent the underlying problem, improving model performance. It includes creating new features from existing ones (like extracting day-of-week from dates), transforming features (scaling, normalization, log transformation), handling categorical variables (one-hot encoding, target encoding), handling text (TF-IDF, word embeddings), and feature selection/extraction. Feature engineering is important because algorithms learn from the features provided; better features lead to better models. It incorporates domain knowledge into the data, can reduce dimensionality, handle missing values, and expose hidden patterns. While deep learning can automatically learn representations, feature engineering remains crucial for most algorithms, especially with limited data. It's often the most time-consuming but impactful part of the machine learning pipeline.",
                "keywords": ["transformation", "domain knowledge", "scaling", "encoding", "text processing", "dimensionality", "missing values", "patterns", "model performance", "pipeline"]
            },
            {
                "question": "What is the difference between bagging and boosting in ensemble learning?",
                "ideal_answer": "Bagging (Bootstrap Aggregating) and boosting are ensemble techniques that combine multiple models to improve performance, but they work differently. Bagging trains models independently in parallel on random subsets of data (bootstrapped samples), often with random feature subsets too. Models vote or average predictions, reducing variance and overfitting. Random Forest is a popular bagging algorithm. Boosting trains models sequentially, with each model focusing on the errors of previous ones. It assigns higher weights to misclassified instances, reducing bias and improving predictions on difficult cases. Popular boosting algorithms include AdaBoost, Gradient Boosting, and XGBoost. Bagging works better with complex, low-bias models prone to overfitting, while boosting works better with simple, high-bias models. Bagging is less prone to overfitting than boosting but may be less powerful for complex relationships.",
                "keywords": ["bootstrap", "parallel", "sequential", "variance", "bias", "Random Forest", "AdaBoost", "Gradient Boosting", "XGBoost", "weights"]
            },
            {
                "question": "How do you approach a data science project from start to finish?",
                "ideal_answer": "A comprehensive data science project approach includes: 1) Problem Definition: Understand business objectives, define success metrics, and formulate the problem. 2) Data Collection: Gather relevant data from various sources, ensuring proper permissions. 3) Data Exploration: Perform EDA to understand distributions, relationships, and anomalies. 4) Data Preparation: Clean data, handle missing values, outliers, and perform feature engineering. 5) Model Selection: Choose appropriate algorithms based on the problem type and data characteristics. 6) Model Training: Train multiple models with cross-validation and hyperparameter tuning. 7) Model Evaluation: Assess performance using appropriate metrics and business criteria. 8) Model Deployment: Implement the model in production with monitoring systems. 9) Communication: Present findings to stakeholders with clear visualizations and explanations. 10) Iteration: Continuously improve the model based on feedback and new data. Throughout, maintain documentation, version control, and ethical considerations.",
                "keywords": ["problem definition", "EDA", "cleaning", "feature engineering", "cross-validation", "hyperparameter tuning", "evaluation", "deployment", "communication", "iteration"]
            }
        ],
        "Web Development": [
            {
                "question": "Explain the difference between client-side and server-side rendering.",
                "ideal_answer": "Client-side rendering (CSR) executes JavaScript in the browser to generate HTML dynamically after the page loads. The server sends a minimal HTML file with JavaScript that fetches data and renders content. CSR offers rich interactivity, reduced server load, and faster subsequent page transitions. However, it has slower initial load times, potential SEO challenges, and requires more client resources. Server-side rendering (SSR) generates complete HTML on the server before sending it to the browser. This provides faster initial page loads, better SEO as content is immediately available to crawlers, and works better on low-powered devices. Drawbacks include higher server load, slower page transitions, and less interactivity without additional JavaScript. Hybrid approaches like Next.js combine both methods with static site generation (SSG) and incremental static regeneration (ISR) for optimal performance.",
                "keywords": ["JavaScript", "HTML", "browser", "server", "SEO", "load times", "interactivity", "Next.js", "static generation", "hybrid"]
            },
            {
                "question": "What are the key principles of responsive web design?",
                "ideal_answer": "Responsive web design (RWD) adapts websites to different screen sizes and devices through several key principles: Fluid grids use relative units (percentages, em, rem) instead of fixed pixels, allowing content to scale proportionally. Flexible images with max-width:100% ensure they don't exceed their containers. Media queries apply different CSS based on device characteristics like screen width. The mobile-first approach starts with designing for mobile devices and progressively enhances for larger screens. Content prioritization displays the most important content first on smaller screens. Touch-friendly interfaces have appropriately sized tap targets. Performance optimization keeps page sizes small for mobile networks. Viewport meta tag ensures proper scaling. Testing across multiple devices is essential. Modern frameworks like Bootstrap and CSS Grid/Flexbox simplify implementation of these principles.",
                "keywords": ["fluid grids", "flexible images", "media queries", "mobile-first", "content prioritization", "touch-friendly", "performance", "viewport", "testing", "frameworks"]
            },
            {
                "question": "Explain the concept of RESTful APIs and their principles.",
                "ideal_answer": "RESTful APIs (Representational State Transfer) are architectural style for designing networked applications based on six guiding principles: Client-server architecture separates concerns, improving portability and scalability. Statelessness means each request contains all information needed, with no client context stored on the server. Cacheability indicates whether responses can be cached to improve performance. Layered system allows for load balancing and security policies. Uniform interface simplifies architecture with four constraints: resource identification in requests, resource manipulation through representations, self-descriptive messages, and hypermedia as the engine of application state (HATEOAS). Code on demand (optional) allows client functionality to be extended by downloading code. RESTful APIs typically use HTTP methods (GET, POST, PUT, DELETE) to perform CRUD operations on resources, with responses usually in JSON or XML format.",
                "keywords": ["client-server", "stateless", "cacheable", "layered", "uniform interface", "HATEOAS", "HTTP methods", "resources", "JSON", "CRUD"]
            },
            {
                "question": "What are the differences between HTTP/1.1, HTTP/2, and HTTP/3?",
                "ideal_answer": "HTTP/1.1 (1997) established persistent connections and request pipelining, but suffered from head-of-line blocking and inefficient header handling. HTTP/2 (2015) introduced binary framing, multiplexing (multiple requests/responses over one connection), header compression (HPACK), server push, and stream prioritization. These improvements significantly reduced latency and improved page load times. However, HTTP/2 still faces head-of-line blocking at the TCP level. HTTP/3 (formerly QUIC) uses UDP instead of TCP with built-in encryption (TLS 1.3), improved connection establishment (0-RTT), independent streams that eliminate head-of-line blocking at the transport level, and better performance on unreliable networks with connection migration. Each version maintains backward compatibility while addressing limitations of previous versions, with adoption depending on server and client support.",
                "keywords": ["persistent connections", "multiplexing", "binary framing", "header compression", "server push", "head-of-line blocking", "UDP", "QUIC", "TLS", "connection migration"]
            },
            {
                "question": "Explain Cross-Origin Resource Sharing (CORS) and its purpose.",
                "ideal_answer": "Cross-Origin Resource Sharing (CORS) is a security feature implemented by browsers that restricts web pages from making requests to domains different from the one that served the original page (same-origin policy). CORS enables servers to specify who can access their resources and how, allowing controlled cross-origin requests. It works through HTTP headers: the browser sends an Origin header with cross-origin requests; the server responds with Access-Control-Allow-Origin specifying allowed origins. For complex requests (non-GET/POST or with custom headers), browsers send a preflight OPTIONS request to check permissions via Access-Control-Allow-Methods and Access-Control-Allow-Headers. Additional headers control credential inclusion and exposed response headers. CORS prevents malicious sites from accessing sensitive data on other domains but can be challenging to configure correctly, requiring proper server-side implementation to balance security and functionality.",
                "keywords": ["same-origin policy", "HTTP headers", "Origin", "Access-Control-Allow-Origin", "preflight", "OPTIONS", "credentials", "security", "browser restriction", "server configuration"]
            },
            {
                "question": "What are web accessibility standards and why are they important?",
                "ideal_answer": "Web accessibility standards, primarily the Web Content Accessibility Guidelines (WCAG), provide a framework for making web content accessible to people with disabilities. WCAG is organized around four principles: Perceivable (information must be presentable to users in ways they can perceive), Operable (interface components must be operable), Understandable (information and operation must be understandable), and Robust (content must be robust enough to work with current and future technologies). Accessibility is important for ethical reasons (digital inclusion), legal compliance (laws like ADA, Section 508, EAA), and business benefits (larger audience, better SEO, improved usability for all). Implementation involves semantic HTML, proper contrast, keyboard navigation, screen reader compatibility, alternative text for images, captions for multimedia, and responsive design. Testing tools include WAVE, axe, and screen readers. The current standard is WCAG 2.1, with levels A, AA, and AAA indicating conformance levels.",
                "keywords": ["WCAG", "perceivable", "operable", "understandable", "robust", "disabilities", "semantic HTML", "screen readers", "alternative text", "keyboard navigation"]
            },
            {
                "question": "Explain the concept of Progressive Web Apps (PWAs) and their key features.",
                "ideal_answer": "Progressive Web Apps (PWAs) are web applications that use modern web capabilities to deliver app-like experiences to users. Key features include: Progressive enhancement works for all users regardless of browser choice. Responsive design functions on any device (desktop, mobile, tablet). Connectivity independence works offline or with poor networks using Service Workers. App-like interactions feel like a native app with app-shell architecture. Fresh content stays up-to-date with Service Worker update process. Safe content served via HTTPS prevents snooping. Discoverable and identifiable as 'applications' thanks to W3C manifests and Service Worker registration. Re-engageable through push notifications. Installable to the home screen without app store. Linkable via URL without complex installation. PWAs combine the best of web and mobile apps: they're searchable, linkable, and deployable, while offering offline functionality, push notifications, and home screen access.",
                "keywords": ["Service Workers", "responsive", "offline", "app-shell", "HTTPS", "manifest", "push notifications", "installable", "home screen", "progressive enhancement"]
            },
            {
                "question": "What are the different ways to optimize website performance?",
                "ideal_answer": "Website performance optimization involves multiple strategies: Minimize HTTP requests by combining files, using CSS sprites, and lazy loading images. Optimize assets by compressing images, minifying CSS/JS/HTML, and using modern formats like WebP. Implement efficient caching with appropriate headers and service workers for offline access. Use Content Delivery Networks (CDNs) to serve assets from geographically closer locations. Enable browser caching with proper cache-control headers. Implement critical CSS rendering path by inlining critical CSS and deferring non-critical resources. Optimize JavaScript execution with code splitting, tree shaking, and asynchronous loading. Reduce server response time through database optimization and efficient hosting. Implement text compression (Gzip/Brotli) for transferred assets. Use responsive images with srcset and sizes attributes. Consider performance budgets and Core Web Vitals metrics (LCP, FID, CLS). Tools like Lighthouse, WebPageTest, and Chrome DevTools help identify opportunities for improvement.",
                "keywords": ["HTTP requests", "minification", "compression", "CDN", "caching", "critical path", "lazy loading", "code splitting", "Core Web Vitals", "responsive images"]
            },
            {
                "question": "Explain the concept of state management in frontend frameworks.",
                "ideal_answer": "State management in frontend frameworks handles the data that determines application behavior and UI rendering. Local component state manages data specific to a component, like form inputs or toggle states. Global/application state manages data shared across components, like user authentication or theme preferences. State management solutions vary by framework: React uses useState/useReducer hooks for local state and Context API, Redux, or Zustand for global state. Vue has reactive data properties and Vuex/Pinia. Angular uses services and NgRx. These solutions address challenges like prop drilling (passing state through multiple components), maintaining a single source of truth, handling asynchronous operations, and ensuring predictable state updates. Modern approaches favor immutability, unidirectional data flow, and separation of state logic from UI components. Effective state management improves maintainability, debugging, and performance by preventing unnecessary re-renders and enabling time-travel debugging.",
                "keywords": ["local state", "global state", "Redux", "Context API", "prop drilling", "immutability", "unidirectional flow", "predictable updates", "debugging", "performance"]
            },
            {
                "question": "What security considerations are important in web development?",
                "ideal_answer": "Web security requires multiple defensive layers: Prevent Cross-Site Scripting (XSS) by sanitizing user input, using Content Security Policy (CSP), and implementing output encoding. Avoid SQL Injection through parameterized queries and ORMs. Protect against Cross-Site Request Forgery (CSRF) with anti-CSRF tokens. Implement proper authentication with secure password storage (using bcrypt/Argon2), multi-factor authentication, and account lockouts. Use secure session management with HTTPOnly and Secure cookie flags. Configure proper CORS policies to control cross-origin requests. Enable HTTPS with TLS 1.2+ and proper certificate management. Implement security headers like Strict-Transport-Security, X-Content-Type-Options, and X-Frame-Options. Practice secure dependency management by keeping libraries updated and using tools like Snyk. Follow the principle of least privilege for user permissions. Conduct regular security testing including penetration testing and code reviews. Stay informed about OWASP Top 10 vulnerabilities and security best practices.",
                "keywords": ["XSS", "CSRF", "SQL Injection", "authentication", "HTTPS", "security headers", "input validation", "dependency management", "OWASP", "least privilege"]
            }
        ],
        "Machine Learning": [
            {
                "question": "Explain the difference between parametric and non-parametric models.",
                "ideal_answer": "Parametric models make strong assumptions about the data distribution and can be described by a fixed number of parameters, regardless of the training data size. Examples include linear regression, logistic regression, and neural networks. They're computationally efficient, require less data, and are easier to interpret, but may underperform if their assumptions don't match the true data distribution. Non-parametric models make fewer assumptions and use a flexible number of parameters that can grow with the training data. Examples include decision trees, random forests, k-nearest neighbors, and kernel methods. They can capture complex relationships and adapt to any function shape, but require more data, are computationally intensive, and may overfit without proper regularization. The choice between parametric and non-parametric models involves trade-offs between computational efficiency, flexibility, interpretability, and the amount of available data. In practice, the best approach often depends on the specific problem, with parametric models working well for simpler relationships and non-parametric models excelling at capturing complex, unknown relationships in the data.",
                "keywords": ["assumptions", "fixed parameters", "flexible", "linear regression", "decision trees", "k-nearest neighbors", "computational efficiency", "interpretability", "data requirements", "complexity"]
            },
            {
                "question": "What is the difference between L1 and L2 regularization?",
                "ideal_answer": "L1 (Lasso) and L2 (Ridge) regularization are techniques to prevent overfitting by adding penalty terms to the loss function. L1 regularization adds the sum of absolute values of coefficients (|w|) to the loss function. It produces sparse models by driving some coefficients exactly to zero, effectively performing feature selection. This makes it useful for high-dimensional data with many irrelevant features. L2 regularization adds the sum of squared coefficients (w²) to the loss function. It shrinks all coefficients toward zero but rarely makes them exactly zero. L2 works better when most features are relevant and for dealing with multicollinearity. Elastic Net combines both approaches. The regularization strength is controlled by a hyperparameter (lambda or alpha) that balances the original loss with the penalty term. L1 has no analytical solution and can be computationally challenging, while L2 has a closed-form solution.",
                "keywords": ["Lasso", "Ridge", "penalty", "sparsity", "feature selection", "shrinkage", "multicollinearity", "Elastic Net", "hyperparameter", "coefficients"]
            },
            {
                "question": "Explain the concept of gradient descent and its variations.",
                "ideal_answer": "Gradient descent is an iterative optimization algorithm that minimizes a function by moving in the direction of steepest descent (negative gradient). In machine learning, it's used to find the model parameters that minimize the loss function. The algorithm updates parameters using the formula: parameter = parameter - learning_rate * gradient. The learning rate controls the step size. Batch gradient descent computes the gradient using the entire dataset, which is computationally expensive but stable. Stochastic gradient descent (SGD) uses a single random sample for each update, making it faster but with noisier convergence. Mini-batch gradient descent strikes a balance by using small batches of samples. Advanced variations include momentum (adds a fraction of the previous update to accelerate convergence), AdaGrad (adapts learning rates per parameter), RMSProp (addresses AdaGrad's diminishing learning rates), and Adam (combines momentum and adaptive learning rates). These variations help overcome challenges like slow convergence, getting stuck in local minima, and selecting appropriate learning rates.",
                "keywords": ["optimization", "loss function", "learning rate", "batch", "stochastic", "mini-batch", "momentum", "AdaGrad", "RMSProp", "Adam"]
            },
            {
                "question": "How do Convolutional Neural Networks (CNNs) work?",
                "ideal_answer": "Convolutional Neural Networks (CNNs) are specialized deep learning architectures designed primarily for image processing. Their structure mimics the visual cortex with layers that automatically extract hierarchical features. The key components include: Convolutional layers apply filters (kernels) that slide across the input, performing element-wise multiplication and summation to detect features like edges, textures, and patterns. Activation functions (typically ReLU) introduce non-linearity. Pooling layers (max or average pooling) reduce spatial dimensions, providing translation invariance and computational efficiency. Fully connected layers at the end perform classification based on the extracted features. CNNs use parameter sharing (same filter applied across the image) and local connectivity (neurons connect to only a small region of the previous layer), dramatically reducing parameters compared to fully connected networks. Modern CNN architectures include AlexNet, VGG, ResNet (with skip connections to address vanishing gradients), Inception (with parallel convolutions), and EfficientNet (optimized for efficiency). CNNs excel at computer vision tasks like image classification, object detection, and segmentation.",
                "keywords": ["convolution", "filters", "kernels", "pooling", "parameter sharing", "local connectivity", "feature extraction", "hierarchical", "translation invariance", "computer vision"]
            },
            {
                "question": "What is transfer learning and when would you use it?",
                "ideal_answer": "Transfer learning is a machine learning technique where a model developed for one task is reused as the starting point for a model on a second task. It leverages knowledge (features, weights) from pre-trained models instead of starting from scratch. The process typically involves taking a pre-trained model (e.g., on ImageNet), removing the final layer(s), adding new layers for the target task, and fine-tuning by either updating all weights or freezing early layers and training only the new ones. Transfer learning is particularly useful when you have limited labeled data for your target task, when the source and target domains are related (sharing similar features), when you need faster development time, or when you lack computational resources for training complex models from scratch. It's widely used in computer vision (using models like ResNet, VGG) and NLP (using models like BERT, GPT). The effectiveness depends on the similarity between source and target tasks and the amount of available target data.",
                "keywords": ["pre-trained", "fine-tuning", "feature extraction", "limited data", "computational efficiency", "domain similarity", "frozen layers", "ImageNet", "BERT", "knowledge transfer"]
            },
            {
                "question": "Explain the concept of ensemble learning and common ensemble methods.",
                "ideal_answer": "Ensemble learning combines multiple models to improve performance beyond what any single model could achieve. It works by reducing variance (bagging), bias (boosting), or both. Common ensemble methods include: Bagging (Bootstrap Aggregating) trains models on random subsets of data with replacement and averages predictions, reducing variance. Random Forest is a popular example that also uses feature randomization. Boosting trains models sequentially, with each focusing on previous models' errors. Examples include AdaBoost (adjusts instance weights), Gradient Boosting (fits new models to residuals), and XGBoost (adds regularization). Stacking combines models by training a meta-learner on their predictions. Voting uses simple averaging (regression) or majority voting (classification) from multiple models. Key advantages include improved accuracy, reduced overfitting, increased stability, and better handling of complex relationships. Disadvantages include increased complexity, computational cost, and reduced interpretability. Diversity among base models is crucial for ensemble effectiveness.",
                "keywords": ["bagging", "boosting", "stacking", "voting", "Random Forest", "AdaBoost", "Gradient Boosting", "XGBoost", "diversity", "meta-learner"]
            },
            {
                "question": "What is the difference between a generative and discriminative model?",
                "ideal_answer": "Discriminative models learn the boundary between classes by modeling the conditional probability P(y|x) - the probability of a label y given features x. They focus on directly predicting the target variable without modeling how the data was generated. Examples include logistic regression, SVMs, neural networks, and decision trees. They typically perform better when there's sufficient labeled data and the decision boundary is the primary interest. Generative models learn the joint probability distribution P(x,y) or P(x|y) - how the data was generated given the class. They can generate new samples and model the underlying data distribution. Examples include Naive Bayes, Hidden Markov Models, Gaussian Mixture Models, GANs, and VAEs. Generative models often work better with limited data, handle missing features well, and provide insights about data structure. However, they may make stronger assumptions about data distribution. The choice depends on available data, the need to generate samples, and whether understanding the data generation process is important.",
                "keywords": ["conditional probability", "joint probability", "decision boundary", "data generation", "Naive Bayes", "GANs", "VAEs", "labeled data", "missing features", "assumptions"]
            },
            {
                "question": "How do you handle imbalanced datasets in machine learning?",
                "ideal_answer": "Handling imbalanced datasets requires multiple approaches: Data-level techniques include oversampling minority classes (random oversampling, SMOTE, ADASYN), undersampling majority classes (random undersampling, Tomek links, NearMiss), or hybrid methods. Algorithm-level approaches involve using algorithms less sensitive to imbalance (tree-based methods), cost-sensitive learning (assigning higher misclassification costs to minority classes), or ensemble methods like balanced random forests. Evaluation metrics should go beyond accuracy to precision, recall, F1-score, ROC-AUC, PR-AUC (better for highly imbalanced data), and confusion matrices. Threshold moving adjusts the classification threshold to optimize for specific metrics. Advanced techniques include anomaly/novelty detection approaches for extreme imbalance, one-class classification, and data augmentation. The best approach depends on the imbalance ratio, dataset size, domain knowledge, and business objectives. Often, combining multiple techniques (like SMOTE with ensemble methods) yields the best results. It's also important to maintain proper cross-validation strategies that preserve class distributions across folds.",
                "keywords": ["oversampling", "undersampling", "SMOTE", "cost-sensitive", "precision-recall", "F1-score", "threshold", "ensemble", "cross-validation", "class distribution"]
            },
            {
                "question": "Explain the concept of explainable AI and its importance.",
                "ideal_answer": "Explainable AI (XAI) refers to methods and techniques that make AI systems' decisions understandable to humans. It addresses the 'black box' problem of complex models like deep neural networks. XAI techniques include: Model-agnostic methods like LIME and SHAP that explain any model's predictions by approximating local behavior. Intrinsically interpretable models like linear regression, decision trees, and rule-based systems. Feature importance measures that quantify each feature's contribution to predictions. Partial dependence plots showing how features affect predictions. Attention mechanisms in neural networks highlighting relevant input parts. Counterfactual explanations describing minimal changes needed to alter predictions. XAI is important for regulatory compliance (GDPR's 'right to explanation'), building user trust, detecting bias and fairness issues, debugging models, enabling human oversight, and facilitating deployment in high-stakes domains like healthcare and finance. The field balances the trade-off between model performance and interpretability, with ongoing research developing methods that maintain accuracy while improving transparency.",
                "keywords": ["black box", "interpretability", "transparency", "LIME", "SHAP", "feature importance", "counterfactual", "trust", "bias", "regulatory compliance"]
            },
            {
                "question": "What are the ethical considerations in machine learning?",
                "ideal_answer": "Ethical considerations in machine learning span multiple dimensions: Bias and fairness issues arise when models discriminate against protected groups due to biased training data or algorithmic design. Addressing this requires diverse data, fairness metrics, and algorithmic debiasing techniques. Privacy concerns include data collection consent, anonymization limitations, and inference attacks. Techniques like differential privacy, federated learning, and secure multi-party computation help protect privacy. Transparency and explainability are essential for understanding model decisions, especially in high-stakes domains. Accountability determines who's responsible for AI systems' actions and requires governance frameworks. Security vulnerabilities like adversarial attacks must be addressed through robust model design. Environmental impact from large-scale training requires energy-efficient algorithms. Social impact considerations include job displacement, filter bubbles, and amplification of societal inequalities. Ethical frameworks like beneficence (doing good), non-maleficence (avoiding harm), autonomy (respecting choices), and justice (fair distribution) guide responsible ML development. Ongoing stakeholder engagement and ethical review processes are crucial throughout the ML lifecycle.",
                "keywords": ["bias", "fairness", "privacy", "transparency", "accountability", "security", "environmental impact", "social impact", "governance", "stakeholder engagement"]
            }
        ]
    }
    return questions

# Function to evaluate answer quality
def evaluate_answer(user_answer, ideal_answer, keywords):
    if not user_answer or user_answer.strip() == "":
        return 0, []
    
    # Preprocess answers
    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(tokens)
    
    processed_user_answer = preprocess_text(user_answer)
    processed_ideal_answer = preprocess_text(ideal_answer)
    
    # Calculate similarity score
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform([processed_ideal_answer, processed_user_answer])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    except:
        similarity = 0
    
    # Check for keywords
    keyword_matches = []
    for keyword in keywords:
        if keyword.lower() in processed_user_answer:
            keyword_matches.append(keyword)
    
    keyword_score = len(keyword_matches) / len(keywords)
    
    # Calculate final score (weighted average)
    final_score = 0.6 * similarity + 0.4 * keyword_score
    
    # Scale to 0-10
    scaled_score = min(10, max(0, final_score * 10))
    
    return scaled_score, keyword_matches

# Function to generate feedback
def generate_feedback(scores, answers, questions, domain):
    total_score = sum(scores)
    avg_score = total_score / len(scores) if scores else 0
    
    # Identify strengths and areas for improvement
    strengths = []
    improvements = []
    
    # Analyze individual question performance
    for i, (score, answer, question) in enumerate(zip(scores, answers, questions)):
        if score >= 7:
            if score >= 9:
                strengths.append(f"Excellent understanding of {question['question'].split('?')[0].lower()}")
            else:
                strengths.append(f"Good grasp of {question['question'].split('?')[0].lower()}")
        else:
            if score <= 3:
                improvements.append(f"Need significant improvement in {question['question'].split('?')[0].lower()}")
            else:
                improvements.append(f"Could strengthen knowledge about {question['question'].split('?')[0].lower()}")
    
    # Analyze overall performance
    if avg_score >= 8:
        overall = f"Overall, you demonstrated strong knowledge in {domain}. Your responses were comprehensive and well-articulated."
        if avg_score >= 9:
            overall += " You showed expert-level understanding of the concepts."
    elif avg_score >= 6:
        overall = f"Overall, you have a good foundation in {domain}, but there's room for improvement in some areas."
    else:
        overall = f"You need to strengthen your knowledge in {domain}. Focus on understanding the fundamental concepts better."
    
    # Limit to top 3 strengths and improvements for clarity
    strengths = strengths[:3]
    improvements = improvements[:3]
    
    # Add general advice based on score patterns
    if max(scores) - min(scores) > 5:
        improvements.append("Your knowledge seems uneven across different topics. Try to build a more consistent understanding.")
    
    if all(score < 5 for score in scores):
        improvements.append(f"Consider taking a structured course or reading more about {domain} fundamentals.")
    
    if all(score > 7 for score in scores):
        strengths.append("You have consistent knowledge across different topics, showing good breadth of understanding.")
    
    return {
        "total_score": total_score,
        "average_score": avg_score,
        "strengths": strengths,
        "improvements": improvements,
        "overall_feedback": overall
    }

# Main function for the interview process
def start_interview():
    st.title("🎯 Mock Interview Assistant")
    
    # Domain selection
    if st.session_state.domain is None:
        st.header("Select Your Interview Domain")
        domains = list(load_questions().keys())
        
        cols = st.columns(len(domains))
        for i, domain in enumerate(domains):
            with cols[i]:
                if st.button(f"{domain}", key=f"domain_{domain}", use_container_width=True):
                    st.session_state.domain = domain
                    st.session_state.current_question = 0
                    st.session_state.answers = []
                    st.session_state.scores = []
                    st.session_state.start_time = time.time()
                    st.session_state.question_times = []
                    st.rerun()
    
    # Interview in progress
    elif not st.session_state.interview_complete:
        questions = load_questions()[st.session_state.domain]
        
        # Display progress
        progress = st.progress(st.session_state.current_question / len(questions))
        st.header(f"{st.session_state.domain} Interview")
        st.subheader(f"Question {st.session_state.current_question + 1} of {len(questions)}")
        
        # Display current question
        current_q = questions[st.session_state.current_question]
        st.markdown(f"**{current_q['question']}**")
        
        # Answer input
        answer = st.text_area("Your Answer:", height=200, key="answer_input")
        
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("Skip", key="skip_button"):
                st.session_state.answers.append("")
                st.session_state.scores.append(0)
                st.session_state.question_times.append(0)
                
                if st.session_state.current_question < len(questions) - 1:
                    st.session_state.current_question += 1
                    st.rerun()
                else:
                    st.session_state.interview_complete = True
                    st.rerun()
        
        with col2:
            if st.button("Submit Answer", key="submit_button", use_container_width=True):
                # Calculate time spent on question
                if st.session_state.current_question == 0:
                    time_spent = time.time() - st.session_state.start_time
                else:
                    time_spent = time.time() - sum(st.session_state.question_times) - st.session_state.start_time
                
                st.session_state.question_times.append(time_spent)
                
                # Evaluate answer
                score, keywords_matched = evaluate_answer(
                    answer, 
                    current_q['ideal_answer'], 
                    current_q['keywords']
                )
                
                st.session_state.answers.append(answer)
                st.session_state.scores.append(score)
                
                if st.session_state.current_question < len(questions) - 1:
                    st.session_state.current_question += 1
                    st.rerun()
                else:
                    st.session_state.interview_complete = True
                    st.rerun()
    
    # Interview complete - show results
    else:
        questions = load_questions()[st.session_state.domain]
        feedback = generate_feedback(
            st.session_state.scores,
            st.session_state.answers,
            questions,
            st.session_state.domain
        )
        
        st.header("Interview Complete! 🎉")
        
        # Display overall score
        st.subheader("Your Results")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            # Create a gauge chart for the score
            fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
            
            avg_score = feedback["average_score"]
            theta = np.linspace(0, 1.5*np.pi, 100)
            r = np.ones_like(theta)
            
            # Background
            ax.plot(theta, r, color='lightgray', linewidth=20, alpha=0.5)
            
            # Score indicator
            score_theta = np.linspace(0, 1.5*np.pi * (avg_score/10), 100)
            score_r = np.ones_like(score_theta)
            
            if avg_score < 4:
                color = 'red'
            elif avg_score < 7:
                color = 'orange'
            else:
                color = 'green'
                
            ax.plot(score_theta, score_r, color=color, linewidth=20, alpha=0.8)
            
            # Remove unnecessary elements
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['polar'].set_visible(False)
            
            # Add score text in the middle
            ax.text(0, 0, f"{avg_score:.1f}/10", fontsize=24, ha='center', va='center', fontweight='bold')
            
            st.pyplot(fig)
        
        with col2:
            # Overall feedback
            st.markdown(f"**Overall Assessment:**")
            st.write(feedback["overall_feedback"])
            
            # Time analysis
            total_time = sum(st.session_state.question_times)
            st.markdown(f"**Total Interview Time:** {total_time:.1f} minutes")
            st.markdown(f"**Average Time per Question:** {total_time/len(questions):.1f} minutes")
        
        # Strengths and improvements
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Strengths")
            for strength in feedback["strengths"]:
                st.markdown(f"✅ {strength}")
        
        with col2:
            st.markdown("### Areas for Improvement")
            for improvement in feedback["improvements"]:
                st.markdown(f"🔍 {improvement}")
        
        # Question breakdown
        st.markdown("### Question Breakdown")
        
        # Create a bar chart for question scores
        fig, ax = plt.subplots(figsize=(10, 5))
        question_nums = [f"Q{i+1}" for i in range(len(questions))]
        
        bars = ax.bar(question_nums, st.session_state.scores, color=[
            'red' if score < 4 else 'orange' if score < 7 else 'green' 
            for score in st.session_state.scores
        ])
        
        ax.set_ylim(0, 10)
        ax.set_ylabel('Score')
        ax.set_title('Performance by Question')
        
        # Add score labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        st.pyplot(fig)
        
        # Detailed question analysis
        with st.expander("View Detailed Question Analysis"):
            for i, (question, answer, score) in enumerate(zip(questions, st.session_state.answers, st.session_state.scores)):
                st.markdown(f"**Question {i+1}:** {question['question']}")
                st.markdown(f"**Your Answer:** {answer if answer else '(Skipped)'}")
                st.markdown(f"**Score:** {score:.1f}/10")
                
                # Score interpretation
                if score >= 8:
                    st.markdown("**Feedback:** Excellent answer that covered most key points.")
                elif score >= 6:
                    st.markdown("**Feedback:** Good answer with some important concepts covered.")
                elif score >= 4:
                    st.markdown("**Feedback:** Basic understanding demonstrated, but missing several key concepts.")
                else:
                    st.markdown("**Feedback:** Significant gaps in understanding. Review this topic thoroughly.")
                
                # Show ideal answer
                with st.expander("View Ideal Answer"):
                    st.write(question['ideal_answer'])
                
                st.markdown("---")
        
        # Restart button
        if st.button("Start New Interview", use_container_width=True):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()

# Run the app
if __name__ == "__main__":
    start_interview()