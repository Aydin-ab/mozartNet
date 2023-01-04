# In this file, I want to define functions that I will use to process the data
import music21 as mu
from contextlib import redirect_stdout
import torch



def read_midi(file, show_log= False):
    """ Read music files in format .mid and returns the stream Object

    Parameters
    ----------
    file : string, path to the .mid file
        Input

    Output
    ----------
    stream : list of the successive chords played by the piano.
    """
    
    print("Loading Music File:",file)

    # Parsing a midi file
    stream = mu.converter.parse(file)

    if show_log :
        with open('stream.log', 'w') as f:
            with redirect_stdout(f):
                stream.show('text')
        with open('stream_flatten.log', 'w') as f:
            with redirect_stdout(f):
                stream.flatten().show('text')

    return stream


def stream_to_sequence(stream) :
    """ Read a stream, filter the piano instrument only and output a list of musical images
    If a musical sequence is a musiic sheet, then a musical image is a vertical slice of this sheet.
    A musical image is composed of the chords played by the right and left hand (shape is [right hand, left hand])
    A chord is composed of 5 notes, possibly null and each note is a 2-D vector composed of its pitch and duration.
    A rest is a special note corresponding to the vector [0, 0]
    It is common for a chord to not be composed of 5 notes, if that's the case, we fill the remaining notes with null notes

    The function returns a tensor of the musical sequence of the stream of shape Nx2x5x2 with N the number of musical images extracted

    Parameters
    ----------
    stream : stream object of the MIDI music


    Output
    ----------
    sequence : list of the successive musical images played by the piano.
    """

    piano = mu.instrument.Piano
    REST = [0, 0] # Represent a rest or null note where nothing is played

    # Looping over all the instruments
    # If we know the midi only contains a piano, then we could simply flatten the stream 
    # for part in midi.flatten() :
    # But it is less interpretable to code this way

    # Musical images (= a vertical slice of the music sheet)
    # An image is of shape 2x5x2 : 2 hands x 5 notes or fingers x (1 pitch + 1 duration)
    sequence = []

    # If you want to see what we iterate on
    #with open('partitionByInstrument.log', 'w') as f:
    #    with redirect_stdout(f):
    #        mu.instrument.partitionByInstrument(stream).show('text')

    for part in mu.instrument.partitionByInstrument(stream):
        if isinstance(part.getInstrument(), piano) :
            # We want to retrieve the musical images.
            # During the loop, the notes are iterated in chronological order
            # If the 2 hands play at same time, then the right hand is iterated first then the left

            # We keep track of the time using a clock iterator
            clock = -1 # In quarter duration unit. We check the clock to know if we must merge left and right hand
            image = [] # Buffer to create the musical images one by one

            for element in part.recurse():
                if not ( isinstance(element, mu.note.Note)
                    or isinstance(element, mu.note.Rest)
                    or isinstance(element, mu.chord.Chord) ) :
                    continue

                
                chord = [] # these are the 5 notes (possibly null) played during this element (note/chord/rest)
                
                # Note
                if isinstance(element, mu.note.Note):
                    pitch = element.pitch.midi # MIDI number of the pitch (12-127). 
                                                # Starts at 12 bcs music21 consider <12 to be nearly never existent
                    duration = element.duration.quarterLength # Duration in quarter length unit
                    note = [pitch, duration]
                    chord.append(note)

                # Rest
                elif isinstance(element, mu.note.Rest) :
                    chord.append(REST)

                # Chord
                elif isinstance(element, mu.chord.Chord):
                    for n in element.notes :
                        pitch = n.pitch.midi
                        duration = n.duration.quarterLength
                        note = [pitch, duration]
                        chord.append(note)
                
                # Fill the chord with null notes if it doesn't have 5 notes already
                while len(chord) < 5 :
                    chord.append(REST)
                    # chord is shape 5x2

                # Should we merge with previous hand ?
                #Check the clock to see if it's right or left hand
                if element.offset == clock :
                    # If same offset, then the previous chord was right hand and is present in image, 
                    # and current chord is left hand. We merge them in the image list
                    image.append(chord) # [right, left]
                    if len(image) == 2 :
                        # I've observed that in some MIDI files, there are "Voice" objects that can cause
                        # 3 chords/note/rest at the same time. 
                        # In my current naive implementation, these current lines of code would create one image from the 2 first chords
                        # And 1 single image of the third chord (here). I couldn't figure out the signification of these "Voices"
                        # Therefore, if that happens, I simply discard the third anomalous chord (this current one)
                        # We ignore this third one by making sure that image contains only 2 images
                        sequence.append(image)
                        image = []
                    else :
                        image = []
                
                else :
                    # Else, element.offset > clock so we are looking at the next musical image's right hand
                    if len(image) == 1 :
                        # If image already had a chord in it. We must fill it with a null left hand then reinitialize
                        left_hand = [REST]*5
                        image.append(left_hand)
                        sequence.append(image)
                        image = []
                    image.append(chord)
                clock = element.offset

    sequence = torch.tensor(sequence)
    return sequence


def sequence_to_batch(sequence, length= 32) :
    """ 
    Return a list of inputs given a music sequence and a list of corresponding target image. 
    An input is a sequence of 'length' consecutive musical images
    
    For the first images, we pad the beginning of the input with NULL images.
    For example given a sequence (I1, I2, .., IN) and a length of 3, 
    the first inputs generated are (0, 0, I1), (0, I1, I2), (I1, I2, I3) etc

    For the last images, we pad the end of the input with NULL images.
    For example, the last inputs generated are (I2, I3, 0), (I3, 0, 0) of targets 0

    The target images are the next consecutive image of the corresponding input

    Parameters
    ----------
    sequence : music sequence to divide in inputs
    length : int = 32 : size of an input. Number of consecutive musical images to take into account


    Output
    ----------
    inputs : list of the consecutive inputs
    targets : list of the corresponding target image of the inputs
    """
    
    inputs = []
    targets = []

    # We create the input-target pair using a sliding window.
    # We initialize the window with NULL images + first image
    NULL_IMAGE = torch.zeros(2,5,2)
    window = [NULL_IMAGE]*length 
    inputs.append(torch.stack(window))
    for image in sequence :
        # At iteration T
        targets.append(torch.tensor(image)) #We append here the target n°T-1
        window.pop(0)
        window.append(image)
        inputs.append(torch.stack(window)) # We append here the input n°T
    
    # In this previous loop, we started with an input of [NULL_IMAGE, .., NULL_IMAGE] and target of sequence[0]
    # This pair can be removed as it's not very useful. 
    inputs.pop(0)
    targets.pop(0)

    # Next we add the last input-target pairs meaning the inputs-target of the form 
    # (IN-5, IN-4, IN-3, IN-2, IN-1, IN, 0, 0, ..., 0)  -> target 0 
    # (IN-4, IN-3, IN-2, IN-1, IN, 0, 0, 0, ..., 0)  -> target 0 
    # (IN-3, IN-2, IN-1, IN, 0, 0, 0, 0, ..., 0)  -> target 0 
    # etc
    # Here window is currently of the form (IN-length+1, ..., IN), let's add his target 0
    for _ in range(length-1) :
        targets.append(torch.tensor(NULL_IMAGE))
        window.pop(0)
        window.append(NULL_IMAGE)
        inputs.append(torch.stack(window))
    
    # We add the last target of the last window
    targets.append(torch.tensor(NULL_IMAGE))


    # Now we need to clean the trailing null input-target at the end 
    # It can happen if the MIDI file is not well written
    for input in reversed(inputs) :
        if not torch.any(input) :
            # If all elements is zeros
            inputs.pop(-1)
            targets.pop(-1)



    inputs, targets = torch.stack(inputs), torch.stack(targets)

    return inputs, targets



def main() :
    #midi_filename = 'DATASET/all/balakir/islamei.mid'
    midi_filename = 'DATASET/mozart_midi/mz_311_1.mid'
    midi_stream = read_midi(midi_filename, show_log= True)
    images = stream_to_sequence(midi_stream)
    inputs, targets = sequence_to_batch(images)


    # Need to tokenize the chords
    # Convert each unique chord as a unique token 0, 1, 2 etc

    # Need to create batches of data given the entire chords dataset
    # X = list of 50 consecutive notes, y = 51th note
    test = 0

if __name__ == "__main__" :
    main()