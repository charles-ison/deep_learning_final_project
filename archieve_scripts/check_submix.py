import os

path = "slakh2100_wav_redux/test/"

# check all the dir under the path
i = 0
for root, dirs, files in os.walk(path):
    if root.split('/')[-1].startswith('Track'):
        # i+=1
        bass_path = os.path.join(root, 'bass')
        if not os.path.exists(bass_path):
            print(root, " no bass folder generated")
            # new_place = "slakh2100_wav_redux/fail_submix/"+root.split('/')[-2]+"/"+root.split('/')[-1]
            # os.rename(root,new_place)
            i+=1
        else:
            bass_bass_path = os.path.join(bass_path, 'bass.wav')
            if not os.path.exists(bass_bass_path):
                print(root, " no bass.wav generated")
                # new_place = "slakh2100_wav_redux/fail_submix/"+root.split('/')[-2]+"/"+root.split('/')[-1]
                # os.rename(root,new_place)
                i+=1
            residual_path = os.path.join(bass_path, 'residuals.wav')
            if not os.path.exists(residual_path):
                print(root, " no residuals.wav generated")
                # new_place = "slakh2100_wav_redux/fail_submix/"+root.split('/')[-2]+"/"+root.split('/')[-1]
                # os.rename(root,new_place)
                i+=1
print(i)
    