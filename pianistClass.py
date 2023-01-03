from dataclasses import dataclass
import pygame


@dataclass
class Pianist:
    # mixer default config
    freq: int = 44100 # audio CD quality
    bitsize: int = -16 # unsigned 16 bit
    channels: int = 2 # 1 is mono, 2 is stereo
    buffer: int = 1024 # number of samples
    volume: float = 0.8 # optional volume 0 to 1.0

    def __post_init__(self) :
        pygame.mixer.init(self.freq, self.bitsize, self.channels, self.buffer)
        pygame.mixer.music.set_volume(self.volume)
        
    def play_music(self, midi_filename):
        '''Stream music_file in a blocking manner'''
        clock = pygame.time.Clock()
        pygame.mixer.music.load(midi_filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            clock.tick(30) # check if playback has finished
        

    def listen_music(self, midi_filename) :
        # listen for interruptions
        try:
            # use the midi file you just saved
            self.play_music(midi_filename)
        except KeyboardInterrupt:
            # if user hits Ctrl/C then exit
            # (works only in console mode)
            FADEOUT= 1000
            pygame.mixer.music.fadeout(FADEOUT)
            pygame.mixer.music.stop()
            raise SystemExit



def main() :
    mozart = Pianist()
    print(mozart)
    midi_filename = 'DATASET/mozart_midi/mz_311_1.mid'
    mozart.listen_music(midi_filename)

if __name__ == "__main__" :
    main()