python  main.py --data_root /datasets_general --affix std --batch_size 1
python  main.py --data_root /datasets_general -e 0.157 -p 'linf' --adv_train --affix 'linf'  --batch_size 1
python  main.py --data_root /datasets_general -e 0.314 -p 'l2' --adv_train --affix 'l2' --batch_size 1
