import os
import numpy as np
import argparse

TOPOLOGIES = ["chain", "ring", "star"]

def create_parser():

    parser = argparse.ArgumentParser(description="Generate a mobility network for SIR model simulation.")

    parser.add_argument("output_folder", help="Folder where the mobility matrix will be saved")

    parser.add_argument("-t", "--topology", dest="topology", default="chain", choices=TOPOLOGIES,
                         help="Mobility network topology")

    parser.add_argument("-n", "--nodes", dest="n", default=9, type=int,
                        help="Number of nodes")

    parser.add_argument("-p", "--population", dest="pop", default=10000, type=int,
                        help="Population of each node")

    parser.add_argument("-d", "--delta", dest="delta", default=0.05, type=float,
                        help="Proportion of moving population")
    return parser

def main():
    parser         = create_parser()
    args           = parser.parse_args()

    topology = args.topology
    N = args.n
    pop = args.pop
    delta = args.delta
    output_folder = args.output_folder

    print(args)

    M = np.zeros((N,N))

    for i in range(N-1):
        M[i,i] = 1-delta
        M[i,i+1] = delta

    if topology == "star":
        M[0, 1:] = delta
        M[0,0] = 1- (delta * (N-1))
        M[N-1,N-1] = 1 - delta
        M[N-1,1] = delta

    elif topology == "ring":
        M[N-1,N-1] = 1-delta
        M[N-1,0] = delta

    elif topology == "chain":
        M[N-1,N-1] = 1

    M *= pop

    print(f"Metapopulation matrix generated for {topology}")
    print(M)

    output_file = os.path.join(output_folder, "mobility_{}_{}_pop".format(topology,N))
    print(f"Saving matrix as {output_file}.npy")
    np.save(output_file, M)

if __name__ == "__main__":
    main()
