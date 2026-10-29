"""Run the ShiftScope study with python -m shiftscope."""
import argparse
from .experiment import run,SEEDS,SIGMAS
from .reporting import write_results
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--seeds",default="7,21,42")
    parser.add_argument("--output",default="results")
    args=parser.parse_args()
    seeds=tuple(int(x.strip()) for x in args.seeds.split(",") if x.strip())
    if not seeds:parser.error("At least one random seed required")
    rows=run(seeds)
    summary=write_results(rows,args.output,seeds,SIGMAS)
    print(f"Saved {len(rows)} model/method/shift measurements to {args.output}.")
    for x in summary:
        if x["sigma"] in (0.,4.5) and x["method"]=="temperature_scaled":
            print(f"{x['model']}: sigma={x['sigma']} accuracy={x['accuracy_mean']:.3f} coverage={x['conformal_coverage_mean']:.3f}")
if __name__=="__main__":main()
