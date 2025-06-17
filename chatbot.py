import random
import json
import pickle
import numpy as np
import nltk
import pandas as pd
import re
from fuzzywuzzy import process
from flask import Flask, render_template, request, jsonify
app = Flask(__name__)


from nltk.stem import WordNetLemmatizer
from keras.models import load_model

bus_data = pd.read_csv('surat_bus.csv')
csv_file2 = r"surat_bus.csv"  # Update with your actual path
df2 = pd.read_csv(csv_file2)
csv_file = r"SURAT5.csv"  # Update with your actual path
df = pd.read_csv(csv_file)

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('BusBuddy_Intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

with open('fare_prediction_model.pkl', 'rb') as file:
    fare_model = pickle.load(file)

def predict_fare(origin, destination):
    data = pd.read_csv("SURAT5.csv")
    data['originStopName'] = data['originStopName'].str.strip().str.lower()
    data['destinationStopName'] = data['destinationStopName'].str.strip().str.lower()
    origin = origin.strip().lower()
    destination = destination.strip().lower()
    fare_row = data[
        (data['originStopName'] == origin) & 
        (data['destinationStopName'] == destination)
    ]
    if not fare_row.empty:
        fare_for_adult= fare_row['fareForAdult'].values[0]
        fare_for_child= fare_row['fareForChild'].values[0]
        return fare_for_child,fare_for_adult
    reverse_fare_row = data[
        (data['originStopName'] == destination) & 
        (data['destinationStopName'] == origin)
    ]
    if not reverse_fare_row.empty:
        fare_for_adult = reverse_fare_row['fareForAdult'].values[0]
        fare_for_child = reverse_fare_row['fareForChild'].values[0]
        print(f"Reverse route found (B to A).")
        return fare_for_child, fare_for_adult
    else:
        input_data = pd.DataFrame({
        'originStopName': [origin],
        'destinationStopName': [destination],
        'travelDistance': [random.randint(1, 5)],
        'stage': [random.choice([1, 2, 3])]  
        })
        try:
            preprocessor = fare_model.named_steps['preprocessor']
            preprocessed_data = preprocessor.transform(input_data)
            dense_data = preprocessed_data.toarray() if hasattr(preprocessed_data, "toarray") else preprocessed_data
            print(f"\nDense Preprocessed Data:\n{dense_data}")
            print(f"Shape of Dense Data: {dense_data.shape}")
            prediction = fare_model.named_steps['regressor'].predict(dense_data)
            if len(prediction) == 1:
                fare_for_child = round(prediction[0][0])
                fare_for_adult = round(prediction[0][1])
                return fare_for_child, fare_for_adult
            else:
                print("Unexpected prediction output format.")
                return None, None
        except Exception as e:
            print(f"\nError during prediction: {str(e)}")
            return None, None

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words (sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    if not results:
        return [{'intent': 'no_match', 'probability': '0'}]
    return [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]

destination_list = df['destinationStopName'].tolist()
origin_list = df['originStopName'].tolist()
combined_list = destination_list + origin_list

def extract_origin_destination(message,combined_list, threshold=80):
    # Define a regex pattern to capture the origin and destination
    pattern = r"from\s+(.+?)\s+to\s+(.+)"
    match = re.search(pattern, message.lower())  # case-insensitive match
    
    if match:
        origin = match.group(1).strip()
        destination = match.group(2).strip()
        closest_origin = process.extractOne(origin, combined_list)
        closest_destination = process.extractOne(destination, combined_list)
        origin_match = closest_origin[0].lower() if closest_origin and closest_origin[1] >= threshold else None
        destination_match = closest_destination[0].lower() if closest_destination and closest_destination[1] >= threshold else None
    
        return origin_match, destination_match
    else:
        return None,None
    

all_stops = []
for col in df2.columns[1:]: 
    all_stops.extend(df2[col].dropna().tolist()) 
combined_list2 = sorted(set(all_stops))

def find_direct_buses(start_stop, end_stop):
    direct_buses = []
    for _, row in bus_data.iterrows():
        stops = row[['Stop 1', 'Stop 2', 'Stop 3', 'Stop 4', 'Stop 5', 'Stop 6', 'Stop 7', 'Stop 8', 'Stop 9', 'Stop 10','Stop 11', 'Stop 12', 'Stop 13', 'Stop 14', 'Stop 15', 'Stop 16', 'Stop 17', 'Stop 18', 'Stop 19', 
                    'Stop 20', 'Stop 21', 'Stop 22', 'Stop 23', 'Stop 24', 'Stop 25', 'Stop 26', 'Stop 27', 'Stop 28','Stop 29', 'Stop 30', 'Stop 31', 'Stop 32', 'Stop 33', 'Stop 34', 'Stop 35', 'Stop 36', 'Stop 37', 
                    'Stop 38', 'Stop 39', 'Stop 40', 'Stop 41', 'Stop 42', 'Stop 43', 'Stop 44', 'Stop 45', 'Stop 46','Stop 47', 'Stop 48', 'Stop 49', 'Stop 50', 'Stop 51']].dropna().str.lower().tolist()
        if start_stop in stops and end_stop in stops:
            direct_buses.append(row['bus no'])
    return direct_buses

@app.route('/direct_buses', methods=['GET'])
def direct_buses_route():
    starting = request.args.get('start_stop', '').strip().lower()
    ending = request.args.get('end_stop', '').strip().lower()
    threshold=80
    closest_origin = process.extractOne(starting, combined_list2)
    closest_destination = process.extractOne(ending, combined_list2)
    start_stop = closest_origin[0].lower() if closest_origin and closest_origin[1] >= threshold else None
    end_stop = closest_destination[0].lower() if closest_destination and closest_destination[1] >= threshold else None
    
    if not start_stop or not end_stop:
        return jsonify({"error": "Please provide both 'start_stop' and 'end_stop' as query parameters."}), 400
    
    direct_buses = find_direct_buses(start_stop, end_stop)
    if direct_buses:
        return jsonify({"direct_buses": direct_buses})
    
    return jsonify({"message": f"No direct buses found from {start_stop} to {end_stop}."})

def find_transfer_routes(start_stop, end_stop):
    transfer_routes = []
    buses_from_start = []
    buses_to_end = []
    
    for _, row in bus_data.iterrows():
        stops = row[['Stop 1', 'Stop 2', 'Stop 3', 'Stop 4', 'Stop 5', 'Stop 6', 'Stop 7', 'Stop 8', 'Stop 9', 'Stop 10', 
                    'Stop 11', 'Stop 12', 'Stop 13', 'Stop 14', 'Stop 15', 'Stop 16', 'Stop 17', 'Stop 18', 'Stop 19', 
                    'Stop 20', 'Stop 21', 'Stop 22', 'Stop 23', 'Stop 24', 'Stop 25', 'Stop 26', 'Stop 27', 'Stop 28', 
                    'Stop 29', 'Stop 30', 'Stop 31', 'Stop 32', 'Stop 33', 'Stop 34', 'Stop 35', 'Stop 36', 'Stop 37', 
                    'Stop 38', 'Stop 39', 'Stop 40', 'Stop 41', 'Stop 42', 'Stop 43', 'Stop 44', 'Stop 45', 'Stop 46', 
                    'Stop 47', 'Stop 48', 'Stop 49', 'Stop 50', 'Stop 51']].dropna().str.lower().tolist()
        
        if start_stop in stops:
            start_index = stops.index(start_stop)
            buses_from_start.append({'bus_num': row['bus no'], 'stops': stops, 'start_index': start_index})
        if end_stop in stops:
            end_index = stops.index(end_stop)
            buses_to_end.append({'bus_num': row['bus no'], 'stops': stops, 'end_index': end_index})
    
    for bus_from_start in buses_from_start:
        for transfer_stop in bus_from_start['stops'][bus_from_start['start_index']+1:]:
            for bus_to_end in buses_to_end:
                if transfer_stop in bus_to_end['stops']:
                    if bus_from_start['bus_num'] != bus_to_end['bus_num']:
                        transfer_index = bus_to_end['stops'].index(transfer_stop)
                        bus_combo = (bus_from_start['bus_num'], bus_to_end['bus_num'])
                        if bus_combo not in [(route['first_bus'], route['second_bus']) for route in transfer_routes]:
                            transfer_routes.append({
                                'first_bus': bus_from_start['bus_num'],
                                'transfer_stop': transfer_stop,
                                'second_bus': bus_to_end['bus_num']
                            })
                        break
    return transfer_routes


@app.route('/transfer_routes', methods=['GET'])
def transfer_routes():
    starting = request.args.get('start_stop', '').strip().lower()
    ending= request.args.get('end_stop', '').strip().lower()
    threshold=80
    closest_origin = process.extractOne(starting, combined_list2)
    closest_destination = process.extractOne(ending, combined_list2)
    start_stop = closest_origin[0].lower() if closest_origin and closest_origin[1] >= threshold else None
    end_stop = closest_destination[0].lower() if closest_destination and closest_destination[1] >= threshold else None

    if not start_stop or not end_stop:
        return jsonify({"error": "Please provide both 'start_stop' and 'end_stop' as query parameters."}), 400
    
    transfer_routes = find_transfer_routes(start_stop, end_stop)
    if transfer_routes:
        return jsonify({"transfer_routes": transfer_routes})

    return jsonify({"message": f"No transfer routes found from {start_stop} to {end_stop}."})


def get_response(intents_list, intents_json, message):
    tag = intents_list[0]['intent']
    if tag == "fare":
        print(f"User message received: {message}")
        if "to" not in message.lower():
            return "Please provide the origin and destination in the format 'fare to go from [origin] to [destination]'."
        origin, destination = extract_origin_destination(message,combined_list)
        if not origin or not destination:
            return "I am not aware of any buses going to these stops"
        print(f"Origin: {origin}, Destination: {destination}")
        fare_for_child, fare_for_adult = predict_fare(origin, destination)
        if fare_for_child is not None and fare_for_adult is not None:
            return f"The fare from {origin} to {destination} is ₹{fare_for_child} for children and ₹{fare_for_adult} for adults."
        else:
            return "Sorry, I couldn't predict the fare at this time. Please try again later."
        
    elif tag == "route_availability":
        print(f"User message received: {message}")
        if "from" not in message.lower() or "to" not in message.lower():
            return "Please provide the origin and destination in the format 'bus route from [origin] to [destination]'."
        origin, destination = extract_origin_destination(message,combined_list)
        if not origin or not destination:
            return "It seems there was an issue extracting the origin and destination. Please ensure the format is correct: 'bus route from [origin] to [destination]'."
        print(f"Origin: {origin}, Destination: {destination}")
        
        # Find the direct buses or transfer routes
        direct_buses = find_direct_buses(origin, destination)
        if direct_buses:
            return f"Direct buses from {origin} to {destination}: {', '.join(map(str, direct_buses))}."
        else:
            transfer_routes = find_transfer_routes(origin, destination)
            if transfer_routes:
                transfer_route_str = "\n".join([f"Take bus {route['first_bus']} from {origin} to {route['transfer_stop']}, "
                                            f"then transfer to bus {route['second_bus']} to reach {destination}." 
                                            for route in transfer_routes])
                fare_child_leg1, fare_adult_leg1 = predict_fare(origin, 'transfer_stop')
                fare_child_leg2, fare_adult_leg2 = predict_fare('transfer_stop', destination)
                total_fare_child = fare_child_leg1 + fare_child_leg2
                total_fare_adult = fare_adult_leg1 + fare_adult_leg2
                return f"Transfer options from {origin} to {destination}: \n{transfer_route_str} \n The fare for adults will be ₹{total_fare_adult} and ₹{total_fare_child} for children"
            else:
                return f"No direct or transfer routes found from {origin} to {destination}."
    else:
        # Handle other intents
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
        return result


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    message = request.form["msg"]
    ints = predict_class(message)
    print(f"Predicted classes: {ints}")  # Debugging line
    res = get_response(ints, intents, message)
    return jsonify({"response": res})


if __name__ == "__main__":
    app.run(debug=True)
