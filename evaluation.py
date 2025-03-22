import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('punkt_tab')
import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')


# Download NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

def preprocess_text(text):
    """Preprocess text for analysis"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def evaluate_answer(user_answer, ideal_answer, keywords):
    """
    Evaluate the quality of a user's answer compared to an ideal answer
    
    Parameters:
    - user_answer: The answer provided by the user
    - ideal_answer: The ideal or model answer
    - keywords: List of important keywords that should be present
    
    Returns:
    - score: A score from 0-10
    - matched_keywords: List of keywords found in the user's answer
    """
    if not user_answer or user_answer.strip() == "":
        return 0, []
    
    # Preprocess answers
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

def generate_feedback(scores, answers, questions, domain):
    """
    Generate comprehensive feedback based on interview performance
    
    Parameters:
    - scores: List of scores for each question
    - answers: List of user answers
    - questions: List of question dictionaries
    - domain: The interview domain
    
    Returns:
    - Dictionary containing feedback information
    """
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