data_dir = "/home/spica/mnt_device/aqi/dev_data/"
out_dir = "/home/spica/mnt_device/aqi/dev_data/timesub/"
for line in range(0,14688):
    f1 = open(data_dir+"Beijing", "r")
    f2 = open(data_dir+"Tianjin","r")
    f3 = open(data_dir+"Huludao","r")
    ls1 = f1.readlines()[line]
    ls2 = f2.readlines()[line]
    ls3 = f3.readlines()[line]
    lss = ls1.split("#")
    f = open(out_dir+lss[-1][:-1],"wb")
    f.write(ls1)
    f.write(ls2)
    f.write(ls3)
    f.close()
    f1.close()
    f2.close()
    f3.close()
