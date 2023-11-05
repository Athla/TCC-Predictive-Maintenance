import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  StandardScaler
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE

def scalar_model():
    data = pd.read_csv('src/predictive_maintenance.csv')

    X = data.drop(['Target', 'Failure Type', 'UDI','Product ID','Type'], axis=1)
    y = data['Target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],), name= "InputLayer"),
        tf.keras.layers.Dense(32, activation='relu', name= "HiddenLayer01"),
        tf.keras.layers.Dense(16, activation='relu', name= "HiddenLayer02"),
        tf.keras.layers.Dense(1, activation='sigmoid', name= "OutputLayer")
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision()])

    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

    loss, accuracy, precision = model.evaluate(X_test, y_test)

    print(f'Test Loss: {loss*100:.2f}%, Test Accuracy: {accuracy*100:.2f}%, Test Precision: {precision*100:.2f}%')
    model.save("./SCALAR_precisa_model.keras")
    tf.keras.utils.plot_model(model, "./model.png", True, False)

def near_miss_model():
    data = pd.read_csv('src/predictive_maintenance.csv')

    X = data.drop(['Target', 'Failure Type', 'UDI','Product ID','Type'], axis=1)
    y = data['Target']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    nr = NearMiss()
    X_train, y_train = nr.fit_resample(X_train, y_train)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],), name= "InputLayer"),
        tf.keras.layers.Dense(32, activation='relu', name= "HiddenLayer01"),
        tf.keras.layers.Dense(16, activation='relu', name= "HiddenLayer02"),
        tf.keras.layers.Dense(1, activation='sigmoid', name= "OutputLayer")
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision()])

    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

    loss, accuracy, precision = model.evaluate(X_test, y_test)

    print(f'Test Loss: {loss*100:.2f}%, Test Accuracy: {accuracy*100:.2f}%, Test Precision: {precision*100:.2f}%')
    model.save("./NearMiss_precise_model.keras")
    tf.keras.utils.plot_model(model, "./model.png", True, False)

def SMOTE_model():
    data = pd.read_csv('src/predictive_maintenance.csv')

    X = data.drop(['Target', 'Failure Type', 'UDI','Product ID','Type'], axis=1)
    y = data['Target']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    sm = SMOTE()
    X_train, y_train = sm.fit_resample(X_train, y_train)


    # Define the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],), name= "InputLayer"),
        tf.keras.layers.Dropout(0.3),  # Add dropout layer
        tf.keras.layers.Dense(64, activation='relu', name= "HiddenLayer01"),
        tf.keras.layers.Dropout(0.3),  # Add dropout layer
        tf.keras.layers.Dense(32, activation='relu', name= "HiddenLayer02"),
        tf.keras.layers.Dropout(0.3),  # Add dropout layer
        tf.keras.layers.Dense(16, activation='relu', name= "HiddenLayer03"),
        tf.keras.layers.Dropout(0.3),  # Add dropout layer
        tf.keras.layers.Dense(1, activation='sigmoid', name= "OutputLayer")
    ])

    # Compile the model with adjusted learning rate
    custom_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # Adjust learning rate
    model.compile(optimizer=custom_optimizer, loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision()])

    # Add early stopping
    # early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model on the SMOTE dataset
    model.fit(X_train, y_train, epochs=50, batch_size=64)
    loss, accuracy, precision = model.evaluate(X_test, y_test)

    print(f'Test Loss: {loss*100:.2f}%, Test Accuracy: {accuracy*100:.2f}%, Test Precision: {precision*100:.2f}%')
    model.save("./SMOTE_precise_model.keras")


def use_model():
    import pandas as pd
    model = tf.keras.models.load_model("./precise_model.keras")
    real_data = pd.read_csv("output/realtime_data.csv")
    X_real = real_data[['air_temp','process_temp','rot_speed','torque','tool_wear']]  
    
    scaler = StandardScaler()
    X_real_scaled = scaler.fit_transform(X_real)

    # Make predictions
    predictions = model.predict(X_real_scaled)

    # Threshold predictions to convert to class labels 
    predictions_class = (predictions > 0.5).astype(int)

    # Print product IDs of any machines predicted to fail
    for i, pred in enumerate(predictions_class):
        if pred == 1:
            print(f"Machine: {real_data['p_id'][i]} predicted to fail")
    
    return

if __name__== "__main__":
    # scalar_model()
    near_miss_model()
    # SMOTE_model()