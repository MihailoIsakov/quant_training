import numpy as np
import keras
import keras.backend as K


def reinit_weights(model, mean=0, var=0.1):
    """
    Reinitializes model weights in order not to have to recompile again.
    """
    weights = model.weights
    
    for w in weights:
        shape = w.shape.as_list()
        value = np.random.normal(mean, var, shape)
        K.set_value(w, value)
        

def quantize(matrix, fraction_bits):
    matrix *= 2**fraction_bits
    matrix = np.round(matrix)
    matrix /= 2**fraction_bits
    return matrix


def quantize_random(matrix, fraction_bits):
    raised = np.round(matrix * 2.0**fraction_bits)
    error = matrix * 2.0 ** fraction_bits - raised
    probs = np.random.rand(matrix.shape)
    raised += (probs < np.abs(error)) * np.sign(error)
    result = raised / 2.0 ** fraction_bits
    return result


def quantize_range(matrix, fraction_bits, scale, stochastic=True):
    # scale the matrix (for cases when the distribution isn't around [-1, 1]
    # shift right by fraction_bits and round
    raised = np.round(matrix * scale * 2.0**(fraction_bits-1))
    
    if stochastic:
        # calculate the quantization error 
        error = matrix * scale * 2.0 ** (fraction_bits-1) - raised
        # add a small fraction to quantize stochastically
        probs = np.random.rand(*matrix.shape)
        raised += (probs < np.abs(error)) * np.sign(error)

    # cut off ends
    raised = np.clip(raised, -2**(fraction_bits-1), 2**(fraction_bits-1)-1)
    # shift quantized values left to original space
    result = raised / 2.0 ** (fraction_bits-1) / scale
    return result


def quantize_range_no_zero(matrix, fraction_bits, scale, stochastic=True):
    # scale the matrix (for cases when the distribution isn't around [-1, 1]
    # shift right by fraction_bits and round
    raised = np.round(matrix * scale * 2.0**(fraction_bits-1))
    
    if stochastic:
        # calculate the quantization error 
        error = matrix * scale * 2.0 ** (fraction_bits-1) - raised
        # add a small fraction to quantize stochastically
        probs = np.random.rand(*matrix.shape)
        raised += (probs < np.abs(error)) * np.sign(error)

    # quantize zeros up or down
    raised += np.sign(matrix) * (raised == 0)

    # cut off ends
    raised = np.clip(raised, -2**(fraction_bits-1), 2**(fraction_bits-1)-1)
    # shift quantized values left to original space
    result = raised / 2.0 ** (fraction_bits-1) / scale

    return result

            
class RangeQuantizeCallback(keras.callbacks.Callback):
    def __init__(self, fraction_bits, scale, stochastic=True, zeros=True):
        self.fraction_bits = fraction_bits
        self.scale = scale
        self.stochastic = stochastic
        self.zeros = zeros
        
    def on_batch_end(self, batch, logs=None):
        # for weight in self.model.weights:
            # K.set_value(weight, quantize_range(K.get_value(weight), self.fraction_bits, self.scale, self.stochastic))
        weights = self.model.get_weights()

        for i in range(len(weights)):
            if not self.zeros:
                weights[i] = quantize_range(weights[i], self.fraction_bits, self.scale, self.stochastic)
            else:
                weights[i] = quantize_range_no_zero(weights[i], self.fraction_bits, self.scale, self.stochastic)

        self.model.set_weights(weights)

