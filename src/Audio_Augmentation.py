class AudioAugmentation:
    _settings = {
        'augmentation': "speed shift pitch".split(),
        
        'speed': [0.7, 0.8, 0.9, 1, 1.3, 1.5, 1.7, 1.9, 2],
        
        'pitch': [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 7, 9, 11, 15, 20],
        
        'shift': {
            'direction': "left right".split(),
            'max_shift': [1, 2, 3, 4, 5]
        },
       
    }

   
    def get_settings():
        return AudioAugmentation._settings

    def __init__(self, config, a_file):
        data = self.read_audio_file(a_file)
        self.start(data, config, a_file)

    def start(self, data, config, a_file):
        if config['augmentation'] == 'speed':

            augmented_data = self.stretch(data, config['speed'])
            
           # return augmented_data
            


           
        elif config['augmentation'] == 'pitch':

            augmented_data = self.pitch(data, 16000, config['pitch'])
            #return augmented_data
            

        

        

        elif config['augmentation'] == 'shift':

            augmented_data = self.shift(data, 16000, config['max_shift'], config['direction'])
            
            
        
        return augmented_data
            

           
# sr= 16000으로 만들어주자
    def read_audio_file(self, a_file):
        input_length = 16000
        data = librosa.core.load(a_file)[0]
        if len(data) > input_length:
            data = data[:input_length]
        else:
            data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
        return data
    

    def write_audio_file(self, file, data, sample_rate=16000):
        librosa.output.write_wav(file, data, sample_rate)

    def plot_time_series(self, data):
        fig = plt.figure(figsize=(14, 8))
        plt.title('Raw wave ')
        plt.ylabel('Amplitude')
        plt.plot(np.linspace(0, 1, len(data)), data)
        plt.show()
        
        
    def shift(self, data, sampling_rate, shift_max, shift_direction):
        shift = np.random.randint(sampling_rate * shift_max)
        if shift_direction == 'right':
            shift = -shift
        elif shift_direction == 'left':
            direction = np.random.randint(0, 2)
            if direction == 1:
                shift = -shift
        augmented_data = np.roll(data, shift)
        # Set to silence for heading/ tailing
        if shift > 0:
            augmented_data[:shift] = 0
        else:
            augmented_data[shift:] = 0
        return augmented_data

    def pitch(self, data, sampling_rate, pitch_factor):
        return librosa.effects.pitch_shift(data, 16000, pitch_factor)

    def stretch(self, data, rate):
        data = librosa.effects.time_stretch(data, rate)
        return data
    
    
####################################################################################################################    
    
    def speedNpitch(data,fac):
    """
    peed and Pitch Tuning.
    """
    # you can change low and high here
        length_change = np.random.uniform(low=2, high = 2.6)
        speed_fac = fac  / length_change # try changing 1.0 to 2.0 ... =D
        tmp = np.interp(np.arange(0,len(data),speed_fac),np.arange(0,len(data)),data)
        minlen = min(data.shape[0], tmp.shape[0])
        data *= 0
        data[0:minlen] = tmp[0:minlen]
        return data
    
    def dyn_change(data):
    """
    Random Value Change.
    """
        dyn_change = np.random.uniform(low=2 ,high=10)  # default low = 1.5, high = 3
        #print(dyn_change)
        return (data / dyn_change)
    
    def HPSS(data):
        
        x = librosa.effects.hpss(data.astype('float64'))
        data=x[1]
        
        return data
    
    def noise(data):
        wn = np.random.randn(len(data))
        augmented_data1 = data + item['noise'] * wn
    # Cast back to same data type
        data = augmented_data1.astype(type(data[0]))
        return data
            
 
    
    