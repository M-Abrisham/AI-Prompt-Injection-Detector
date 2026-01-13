    #Train the Model
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

FEATURES_PATH = "data/processed/features.pkl"
MODEL_PATH = "data/processed/model.pkl"

def train_model():
    print(" Training...")

    # load Binary data
    pkl_File = open(FEATURES_PATH, "rb")
    X, y = pickle.load(pkl_File)
    pkl_File.close()


    # Split data: Test + Training 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    clf = LogisticRegression(max_iter=10000, random_state=0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f" Accuracy: {acc * 100:.2f}%")
    

    # Save model
    pkl_File = open(MODEL_PATH, "wb")
    pickle.dump(clf, pkl_File)
    pkl_File.close()

    print("✅ Classifier is Successfully trained and saved")

if __name__ == "__main__":
    train_model()