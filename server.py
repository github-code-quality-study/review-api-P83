import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server
 
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

adj_noun_pairs_count = {}
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

reviews = pd.read_csv('data/reviews.csv').to_dict('records')

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        self.valid_locations = [
            'Albuquerque, New Mexico', 'Carlsbad, California', 'Chula Vista, California', 
            'Colorado Springs, Colorado', 'Denver, Colorado', 'El Cajon, California', 
            'El Paso, Texas', 'Escondido, California', 'Fresno, California', 'La Mesa, California', 
            'Las Vegas, Nevada', 'Los Angeles, California', 'Oceanside, California', 
            'Phoenix, Arizona', 'Sacramento, California', 'Salt Lake City, Utah', 
            'San Diego, California', 'Tucson, Arizona'
        ]


    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        """
        The environ parameter is a dictionary containing some useful
        HTTP request information such as: REQUEST_METHOD, CONTENT_LENGTH, QUERY_STRING,
        PATH_INFO, CONTENT_TYPE, etc.
        """

        if environ["REQUEST_METHOD"] == "GET":
            # Create the response body from the reviews and convert to a JSON byte string
            
            # Write your code here
            query = parse_qs(environ['QUERY_STRING'])
            location  = query.get('location', [None])[0]
            start_date = query.get('start_date', [None])[0]
            end_date = query.get('end_date', [None])[0]

            #filter reviews
            filtered_reviews = []
            for review in reviews:
                if location and review['Location'] !=location:
                    continue

                review_date = datetime.strptime(review['Timestamp'], "%Y-%m-%d %H:%M:%S")

                if start_date and review_date < datetime.strptime(start_date, "%Y-%m-%d"):
                    continue
                if end_date and review_date > datetime.strptime(end_date, "%Y-%m-%d"):
                    continue

                sentiment = self.analyze_sentiment(review['ReviewBody'])
                review_with_sentiment = {
                    "ReviewId": review["ReviewId"],
                    "ReviewBody": review["ReviewBody"],
                    "Location": review["Location"],
                    "Timestamp": review["Timestamp"],
                    "sentiment": sentiment
                }
                filtered_reviews.append(review_with_sentiment)

            #sort by compound sentiment score
            filtered_reviews.sort(key=lambda X: X["sentiment"]["compound"], reverse=True)
            
            #response body
            response_body=json.dumps(filtered_reviews, indent=2).encode("utf-8")

            # Set the appropriate response headers
            start_response("200 OK", [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(response_body)))
             ])
            
            return [response_body]


        if environ["REQUEST_METHOD"] == "POST":
            # Write your code here
            try:
                content_length = int(environ.get('CONTENT_LENGTH', 0))
                post_body = environ['wsgi.input'].read(content_length).decode('utf-8')
                post_data = parse_qs(post_body)

                review_body = post_data.get('ReviewBody', [''])[0]
                location = post_data.get('Location', [''])[0]

                # Validate that both ReviewBody and Location are provided
                if not review_body or not location:
                    start_response("400 Bad Request", [("Content-Type", "application/json")])
                    return [json.dumps({"error": "ReviewBody and Location are required"}).encode("utf-8")]

                # Validate that the Location is in the list of valid locations
                if location not in self.valid_locations:
                    start_response("400 Bad Request", [("Content-Type", "application/json")])
                    return [json.dumps({"error": "Invalid Location"}).encode("utf-8")]

                new_review = {
                    "ReviewId": str(uuid.uuid4()),
                    "ReviewBody": review_body,
                    "Location": location,
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

                reviews.append(new_review)
                response_body = json.dumps(new_review, indent=2).encode("utf-8")
                start_response("201 Created", [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(response_body)))
                ])
                return [response_body]
        
            except Exception as e:
                start_response("500 Internal Server Error", [("Content-Type", "application/json")])
                return [json.dumps({"error": str(e)}).encode("utf-8")]

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = os.environ.get('PORT', 8000)
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()