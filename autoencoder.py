from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras import regularizers


def train_autoencoder(X, encoding_dim=16, epochs=50, batch_size=32):
    input_dim = X.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(10 * input_dim, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(input_layer)
    encoded = Dropout(0.3)(encoded)
    encoded = Dense(encoding_dim, activation='relu')(encoded)
    decoded = Dense(10 * input_dim, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)

    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(X, X, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1)

    encoder = Model(input_layer, encoded)
    return encoder, autoencoder



