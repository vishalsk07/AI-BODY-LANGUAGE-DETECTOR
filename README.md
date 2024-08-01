# Body-Language-Decoder

This is a project which predicts your body language in real time and outputs one of three classes the model was trained on : `Happy`
`Sad`
`Victorius`

## How was the project built?  (A high level overview)
* As the project predicts the body language from one of the three classes. Coordinates of body postures were collected in real-time using OpenCV library, with the help of mediapipe library.

* The body landmarks were collected in a csv format. 

* Now it was time for some training. The machine learning library used here was scikit-learn. 

* Created a pipeline  where the data was trained on `LogisticRegression`,`RidgeClassifier`,`RandomForestClassifier`,`GradientBoostingClassifier`

* Each model was evaluated and the best model was picked.

* At the model predicted the body language in real-time.


## Usage Example

## Cloning and Running
To run this project locally, follow these steps:

1. Clone the repository:
   
<pre>
git clone https://github.com/Rhythm1821/Body-Language-Decoder.git
</pre>

2. Navigate to the project directory:

<pre>
cd Body-Language-Decoder
</pre>

3. Install the required dependencies:
<pre>
pip install -r requirements.txt
</pre>
   
4. Run the real-time face detection Python file:
<pre>
python3 main.py
</pre>

## Project Structure

- `data`: Contains the CSV file where body pose coordinates for training are stored.
- `model`: Houses the trained machine learning model in the form of a pickle file.
- `bodylanguage_decoder.ipynb`: A Jupyter Notebook providing detailed data preprocessing, model training, and evaluation steps.
- `main.py`: The main Python script for real-time body language prediction using the trained model.
- `README.md`: The documentation you're currently reading.
- `requirements.txt`: A file listing the required libraries for easy installation.
