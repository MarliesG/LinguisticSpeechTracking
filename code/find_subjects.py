# 2/10 trials were randomly chosen for each participant and presented in speech-weighted-noise
# aka make sure we are working with participants listening to audiobook 1 in silence
import os

directory_sparrkulee = '/home/luna.kuleuven.be/u0123908/sparrkulee'

subject_folders = [elem for elem in os.listdir(directory_sparrkulee) if elem.startswith('sub-')]

noise_text = "<new_value>5</new_value><succeeded>true</succeeded><description>SNR</description></entry>"

subjects_of_interest = []
for subject in subject_folders:
    session_folder = os.listdir(os.path.join(directory_sparrkulee, subject))
    assert len(session_folder) == 1
    current_subject_folder = os.path.join(directory_sparrkulee, subject, session_folder[0], 'eeg')

    # find apr files
    apr_files = [file for file in os.listdir(os.path.join(current_subject_folder)) if '.apr' in file]
    for apr in apr_files:
        with open(os.path.join(current_subject_folder, apr)) as f:
            contents = f.readlines()

        # remove all spaces (to make sure no formatting issues)
        contents = ''.join(contents)
        contents = contents.replace(' ', '')
        contents = contents.replace('\n', '')

        if noise_text in contents:
            continue

        if 'experiment_file="audiobook_1.xml">' in contents or '_audiobook_1.apx">' in contents:
            subjects_of_interest.append(subject)

print(subjects_of_interest)
# this results: 'sub-001', 'sub-002', 'sub-003', 'sub-005', 'sub-007', 'sub-008', 'sub-009', 'sub-010', 'sub-011',
# 'sub-012', 'sub-013', 'sub-015', 'sub-018', 'sub-019', 'sub-024', 'sub-026', 'sub-027', 'sub-028', 'sub-029', 'sub-030',
# 'sub-031', 'sub-032', 'sub-034', 'sub-035', 'sub-036', 'sub-037', 'sub-038', 'sub-039', 'sub-040', 'sub-041', 'sub-042',
# 'sub-043', 'sub-044', 'sub-045', 'sub-046', 'sub-047', 'sub-048', 'sub-049', 'sub-050', 'sub-051', 'sub-052', 'sub-053',
# 'sub-054', 'sub-055', 'sub-056', 'sub-057', 'sub-058', 'sub-059', 'sub-060', 'sub-061', 'sub-062', 'sub-063', 'sub-064',
# 'sub-065', 'sub-066', 'sub-067', 'sub-068', 'sub-069', 'sub-070', 'sub-071', 'sub-072', 'sub-073', 'sub-074', 'sub-075',
# 'sub-076', 'sub-078', 'sub-079', 'sub-080', 'sub-081', 'sub-082', 'sub-083', 'sub-084', 'sub-085'

