import qnnpy.functions.functions as qf
import qnnpy.instruments.lakeshore336 as ctl
def main():
    inst = ctl.Lakeshore336("GPIB0::12::INSTR")
    print(inst.read_temp(channel="A"))

if __name__ == "__main__":
    main()
