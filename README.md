# mozartNet
Let's train a model to generate piano music inspired by Mozart. Can it complete Mozart's unfinished requiem ?


# Code definition

I'll explain part of the code here
- *pianistClass.py* this class instanciate a pianist that can play MIDI audio files. Hopefully this pianist will play better than Mozart one day.



# Data definition

The main novelty here is our music data modelization.
Let's define our modelization :

- A **note** is a 2-D vector composed of 
    - A **pitch** that takes value between 1 (*low* pitch) and 127 (*high* pitch). On a piano, the keyboard is represented between the number 21 and 108. A **rest** will have a pitch of *0*
    - A **duration** that takes continuous value between 0 (*no duration* and 32 (called a *maxima*). A **rest** will have a duration of *0*

- A **chord** is a 5x2 matrix composed of 5 *notes* which are the 5 notes that a single hand can play at maximum.

- A **musical image** is a 2x5x2 matrix composed of 2 *chords*. One chord represents the left hand, the other one represents the right hand. 

- A **music sequence** is a temporal succession of *musical images*. It can be represented by a music sheet where each image is a vertical slice of the sheet. If the sequence is of length S then the *music sequence* is a tensor of shape Sx2x5x2


Our goal is to generate music. We will have as an input a *music sequence* of length 32 and we will have as an output a *musical image*. Then we will slide that musical image inside the sequence and infer the next image again.


# NOTE

In this representation there is an inherent symmetry between the hands : Suppose the left hand plays the chord A and the right hand plays the chord B. The music stays the same no matter if we the hands play each other's chord (e.g the left hand plays B and right hand plays A). So a musical image tensor gives the same information than if we permute the 1st dimension (the 2 hands)