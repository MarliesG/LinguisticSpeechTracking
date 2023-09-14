# Linguistic Speech Tracking by our Brain
This repository introduces a framework for assessing the brain's ability to track linguistic elements while perceiving continuous speech. When we listen to speech, our brain time locks to the rhythm of specific speech features, a phenomenon known as neural tracking (Brodbeck & Simon, 2020). Traditionally, studies examining neural tracking have concentrated on features derived from speech acoustics. However, this workshop promotes the utilization of linguistic speech features—those that capture speech content—for determining linguistic speech tracking.
Linguistic speech features quantify the level of novel linguistic information within phonemes or words. An example of such a feature is "word surprisal," which evokes a negative brain response approximately 400 ms after word onset. This aligns with findings from studies investigating the N400 event-related brain potential (ERP) response, which is usually explored in controlled sentence or word contexts. This suggests that linguistic tracking facilitates the exploration of higher-level language processing using ecologically valid continuous speech stimuli.
Nonetheless, a challenge emerges: linguistic features correlate with acoustic features. Daube et al. (2019) demonstrated that acoustic speech features can account for apparent responses to linguistic phoneme categories. Consequently, without accounting for acoustic properties, analyses of speech tracking may yield spurious significant linguistic speech tracking findings. To address this, we will focus on a subtraction-based approach to determine linguistic tracking while controlling for speech acoustics.
Linguistic speech tracking opens avenues to comprehend whether and how individuals understand speech. The repository introduces the methodology for determining linguistic speech tracking and how to create linguistic speech features. 

For this repository, we used a subset of [the SparrKULee dataset](https://www.biorxiv.org/content/10.1101/2023.07.24.550310v1). This dataset contains EEG data of 85 young subjects who listen to continuous speech. 

# Are you also into linguistic speech tracking? 
Yesss, me too! During my PhD, it took me quite some struggle to develop this framework and discover all the tools to create these linguistic features. After 5 years, I'm really happy to share it with you (to spare you from some mental breakdowns that I had) and to help you accelerate science. If you use code, I would appreciate your citing my work :) 

# My work
Gillis, M., Vanthornhout, J., Simon, J. Z., Francart, T., & Brodbeck, C. (2021). Neural markers of speech comprehension: measuring EEG tracking of linguistic speech representations, controlling the speech acoustics. Journal of Neuroscience, 41(50), 10316-10329.

Verschueren, E., Gillis, M., Decruy, L., Vanthornhout, J., & Francart, T. (2022). Speech understanding oppositely affects acoustic and linguistic neural tracking in a speech rate manipulation paradigm. Journal of Neuroscience, 42(39), 7442-7453.

Gillis, M., Vanthornhout, J., & Francart, T. (2023). Heard or understood? Neural tracking of language features in a comprehensible story, an incomprehensible story and a word list. eneuro, 10(7).

Gillis, M., Kries, J., Vandermosten, M., & Francart, T. (2023). Neural tracking of linguistic and acoustic speech representations decreases with advancing age. NeuroImage, 267, 119841.
