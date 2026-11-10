#!/usr/bin/env bash
# Reproduces the LoGoS-GPO runs of the paper and the project page, with the arguments of the reported runs.
#   bash scripts/reproduce.sh [table|extra|hetero|sweeps|timing|flops|apps|all]   (default: all)
# Outputs go to $ROOT (default runs/) as <case>/s<seed>/{op,logos,gpo,svgp,logos_hetero}, the layout read by
#   python experiments/paper_figs.py --which all --root runs
# op: stage 1 (train_operator.py); logos: LoGoS-GPO (train_logos_vnngp.py); gpo / svgp: baselines;
# logos_hetero: LoGoS-GPO with heteroscedastic noise (--noise feature). Runs need one CUDA GPU each and resume
# from their last checkpoint when restarted. Data: see data/data.md.
set -eu
cd "$(dirname "$0")/.."
PY=${PY:-python}
ROOT=${ROOT:-runs}
WHAT=${1:-all}

table() {   # Table 3: paper wavelet depth and wavelet level 2 rows, three seeds each
    # burgers
    # seed 0
    $PY experiments/train_operator.py --data burgers --ntrain 1900 --nval 48 --ntest 100 --tail-test --modes 16 --batch 20 --seed 0 --out $ROOT/burgers/s0/op --arch wno --level 8 --wavelet db6 --h1 0.1 --aug shift
    $PY experiments/train_logos_vnngp.py --init $ROOT/burgers/s0/op --out $ROOT/burgers/s0/logos --lr-op 0.0001 --freeze-frac 0.0 --seed 0
    $PY experiments/train_gpo.py --init $ROOT/burgers/s0/op --out $ROOT/burgers/s0/gpo --seed 0
    $PY experiments/train_logos_vnngp.py --init $ROOT/burgers/s0/op --out $ROOT/burgers/s0/svgp --freeze-frac 0.0 --seed 0 --svgp
    # seed 1
    $PY experiments/train_operator.py --data burgers --ntrain 1900 --nval 48 --ntest 100 --modes 16 --batch 20 --seed 1 --out $ROOT/burgers/s1/op --arch wno --level 8 --wavelet db6 --h1 0.1 --aug shift
    $PY experiments/train_logos_vnngp.py --init $ROOT/burgers/s1/op --out $ROOT/burgers/s1/logos --lr-op 0.0001 --freeze-frac 0.0 --seed 0
    $PY experiments/train_gpo.py --init $ROOT/burgers/s1/op --out $ROOT/burgers/s1/gpo --seed 1
    $PY experiments/train_logos_vnngp.py --init $ROOT/burgers/s1/op --out $ROOT/burgers/s1/svgp --freeze-frac 0.0 --seed 1 --svgp
    # seed 2
    $PY experiments/train_operator.py --data burgers --ntrain 1900 --nval 48 --ntest 100 --modes 16 --batch 20 --seed 2 --out $ROOT/burgers/s2/op --arch wno --level 8 --wavelet db6 --h1 0.1 --aug shift
    $PY experiments/train_logos_vnngp.py --init $ROOT/burgers/s2/op --out $ROOT/burgers/s2/logos --lr-op 0.0001 --freeze-frac 0.0 --seed 0
    $PY experiments/train_gpo.py --init $ROOT/burgers/s2/op --out $ROOT/burgers/s2/gpo --seed 2
    $PY experiments/train_logos_vnngp.py --init $ROOT/burgers/s2/op --out $ROOT/burgers/s2/svgp --freeze-frac 0.0 --seed 2 --svgp
    # advection
    # seed 0
    $PY experiments/train_operator.py --data advection_paper --ntrain 2000 --ntest 100 --modes 32 --seed 0 --out $ROOT/advection/s0/op --arch ewno --level 3 --h1 0.1 --aug shift
    $PY experiments/train_logos_vnngp.py --init $ROOT/advection/s0/op --out $ROOT/advection/s0/logos --batch 32 --lr-op 0.0001 --freeze-frac 0.0 --seed 0
    $PY experiments/train_gpo.py --init $ROOT/advection/s0/op --out $ROOT/advection/s0/gpo --seed 0
    $PY experiments/train_logos_vnngp.py --init $ROOT/advection/s0/op --out $ROOT/advection/s0/svgp --batch 32 --freeze-frac 0.0 --seed 0 --svgp
    # seed 1
    $PY experiments/train_operator.py --data advection_paper --ntrain 2000 --ntest 100 --modes 32 --seed 1 --out $ROOT/advection/s1/op --arch ewno --level 3 --h1 0.1 --aug shift
    $PY experiments/train_logos_vnngp.py --init $ROOT/advection/s1/op --out $ROOT/advection/s1/logos --batch 32 --lr-op 0.0001 --freeze-frac 0.0 --seed 0
    $PY experiments/train_gpo.py --init $ROOT/advection/s1/op --out $ROOT/advection/s1/gpo --seed 1
    $PY experiments/train_logos_vnngp.py --init $ROOT/advection/s1/op --out $ROOT/advection/s1/svgp --batch 32 --freeze-frac 0.0 --seed 1 --svgp
    # seed 2
    $PY experiments/train_operator.py --data advection_paper --ntrain 2000 --ntest 100 --modes 32 --seed 2 --out $ROOT/advection/s2/op --arch ewno --level 3 --h1 0.1 --aug shift
    $PY experiments/train_logos_vnngp.py --init $ROOT/advection/s2/op --out $ROOT/advection/s2/logos --batch 32 --lr-op 0.0001 --freeze-frac 0.0 --seed 0
    $PY experiments/train_gpo.py --init $ROOT/advection/s2/op --out $ROOT/advection/s2/gpo --seed 2
    $PY experiments/train_logos_vnngp.py --init $ROOT/advection/s2/op --out $ROOT/advection/s2/svgp --batch 32 --freeze-frac 0.0 --seed 2 --svgp
    # darcy_notch
    for s in 0 1 2; do
        $PY experiments/train_operator.py --data darcy_notch --ntrain 1500 --ntest 100 --batch 16 --lr 0.0005 --seed $s --out $ROOT/darcy_notch/s$s/op --arch wno --level 3 --wavelet db6 --h1 0.1
        $PY experiments/train_logos_vnngp.py --init $ROOT/darcy_notch/s$s/op --out $ROOT/darcy_notch/s$s/logos --epochs 150 --batch 16 --seed $s
        $PY experiments/train_gpo.py --init $ROOT/darcy_notch/s$s/op --out $ROOT/darcy_notch/s$s/gpo --seed $s
        $PY experiments/train_logos_vnngp.py --init $ROOT/darcy_notch/s$s/op --out $ROOT/darcy_notch/s$s/svgp --epochs 150 --batch 16 --freeze-frac 0.0 --seed $s --svgp
    done
    # navier_stokes
    # seed 0
    $PY experiments/train_operator.py --data ns_forcing --ntrain 3000 --ntest 100 --width 48 --epochs 300 --lr 0.0003 --seed 0 --out $ROOT/navier_stokes/s0/op --arch wnof --wavelet db6 --h1 0.1
    $PY experiments/train_logos_vnngp.py --init $ROOT/navier_stokes/s0/op --out $ROOT/navier_stokes/s0/logos --epochs 100 --batch 16 --seed 0
    $PY experiments/train_gpo.py --init $ROOT/navier_stokes/s0/op --out $ROOT/navier_stokes/s0/gpo --seed 0
    $PY experiments/train_logos_vnngp.py --init $ROOT/navier_stokes/s0/op --out $ROOT/navier_stokes/s0/svgp --epochs 100 --batch 16 --lr 0.0005 --freeze-frac 0.0 --seed 0 --svgp
    # seed 1
    $PY experiments/train_operator.py --data ns_forcing --ntrain 3000 --ntest 100 --width 48 --epochs 300 --lr 0.0003 --seed 1 --out $ROOT/navier_stokes/s1/op --arch wnof --wavelet db6 --h1 0.1
    $PY experiments/train_logos_vnngp.py --init $ROOT/navier_stokes/s1/op --out $ROOT/navier_stokes/s1/logos --epochs 100 --batch 16 --seed 1
    $PY experiments/train_gpo.py --init $ROOT/navier_stokes/s1/op --out $ROOT/navier_stokes/s1/gpo --seed 1
    $PY experiments/train_logos_vnngp.py --init $ROOT/navier_stokes/s1/op --out $ROOT/navier_stokes/s1/svgp --epochs 100 --batch 16 --freeze-frac 0.0 --seed 1 --svgp
    # seed 2
    $PY experiments/train_operator.py --data ns_forcing --ntrain 3000 --ntest 100 --width 48 --epochs 300 --lr 0.0003 --seed 2 --out $ROOT/navier_stokes/s2/op --arch wnof --wavelet db6 --h1 0.1
    $PY experiments/train_logos_vnngp.py --init $ROOT/navier_stokes/s2/op --out $ROOT/navier_stokes/s2/logos --epochs 100 --batch 16 --seed 2
    $PY experiments/train_gpo.py --init $ROOT/navier_stokes/s2/op --out $ROOT/navier_stokes/s2/gpo --seed 2
    $PY experiments/train_logos_vnngp.py --init $ROOT/navier_stokes/s2/op --out $ROOT/navier_stokes/s2/svgp --epochs 100 --batch 16 --freeze-frac 0.0 --seed 2 --svgp
    # darcy_notch_l2
    for s in 0 1 2; do
        $PY experiments/train_operator.py --data darcy_notch --ntrain 1500 --ntest 100 --epochs 300 --batch 16 --lr 0.0005 --seed $s --out $ROOT/darcy_notch_l2/s$s/op --arch wno --level 2 --wavelet db6 --h1 0.1
        $PY experiments/train_logos_vnngp.py --init $ROOT/darcy_notch_l2/s$s/op --out $ROOT/darcy_notch_l2/s$s/logos --epochs 150 --batch 16 --seed $s
        $PY experiments/train_gpo.py --init $ROOT/darcy_notch_l2/s$s/op --out $ROOT/darcy_notch_l2/s$s/gpo --seed $s
    done
    # navier_stokes_l2
    for s in 0 1 2; do
        $PY experiments/train_operator.py --data ns_forcing --ntrain 3000 --ntest 100 --width 48 --epochs 200 --lr 0.0003 --seed $s --out $ROOT/navier_stokes_l2/s$s/op --arch wnof --level 2 --wavelet db6 --h1 0.1
        $PY experiments/train_logos_vnngp.py --init $ROOT/navier_stokes_l2/s$s/op --out $ROOT/navier_stokes_l2/s$s/logos --epochs 100 --batch 16 --seed $s
        $PY experiments/train_gpo.py --init $ROOT/navier_stokes_l2/s$s/op --out $ROOT/navier_stokes_l2/s$s/gpo --seed $s
    done
}

extra() {   # further benchmarks (Darcy, airfoil) and the mean-operator comparison on Navier-Stokes
    # darcy
    for s in 0 1 2; do
        $PY experiments/train_operator.py --data darcy_std --ntrain 1000 --modes 12 12 --batch 20 --seed $s --out $ROOT/darcy/s$s/op --arch wnof --level 2 --wavelet db6 --h1 0.1 --aug d4
        $PY experiments/train_logos_vnngp.py --init $ROOT/darcy/s$s/op --out $ROOT/darcy/s$s/logos --epochs 100 --batch 16 --seed $s
        $PY experiments/train_gpo.py --init $ROOT/darcy/s$s/op --out $ROOT/darcy/s$s/gpo --seed $s
        $PY experiments/train_logos_vnngp.py --init $ROOT/darcy/s$s/op --out $ROOT/darcy/s$s/svgp --epochs 100 --batch 16 --freeze-frac 0.0 --seed $s --svgp
    done
    # airfoil
    for s in 0 1 2; do
        $PY experiments/train_operator.py --data airfoil --ntrain 1000 --nval 100 --modes 12 12 --batch 20 --seed $s --out $ROOT/airfoil/s$s/op --arch wnof --level 2 --wavelet db6 --h1 0.1
        $PY experiments/train_logos_vnngp.py --init $ROOT/airfoil/s$s/op --out $ROOT/airfoil/s$s/logos --epochs 100 --batch 16 --seed $s
    done
    # navier_stokes_wno
    # seed 0
    $PY experiments/train_operator.py --data ns_forcing --ntrain 3000 --ntest 100 --epochs 300 --lr 0.0005 --seed 0 --out $ROOT/navier_stokes_wno/s0/op --arch wno --wavelet db6 --h1 0.1
    $PY experiments/train_logos_vnngp.py --init $ROOT/navier_stokes_wno/s0/op --out $ROOT/navier_stokes_wno/s0/logos --epochs 100 --batch 16 --seed 0
    # navier_stokes_ewno
    # seed 0
    $PY experiments/train_operator.py --data ns_forcing --ntrain 3000 --ntest 100 --modes 12 12 --width 48 --epochs 300 --lr 0.0005 --seed 0 --out $ROOT/navier_stokes_ewno/s0/op --arch ewno --level 2 --h1 0.1
    $PY experiments/train_logos_vnngp.py --init $ROOT/navier_stokes_ewno/s0/op --out $ROOT/navier_stokes_ewno/s0/logos --epochs 100 --batch 16 --lr-op 0.0001 --freeze-frac 0.0 --seed 0
}

hetero() {   # heteroscedastic noise (--noise feature) on the stage-1 operators above
    $PY experiments/train_logos_vnngp.py --init $ROOT/burgers/s0/op --out $ROOT/burgers/s0/logos_hetero --lr-op 0.0001 --seed 0 --noise feature
    $PY experiments/train_logos_vnngp.py --init $ROOT/advection/s1/op --out $ROOT/advection/s1/logos_hetero --batch 32 --lr-op 0.0001 --freeze-frac 0.0 --seed 0 --noise feature
    $PY experiments/train_logos_vnngp.py --init $ROOT/darcy_notch_l2/s0/op --out $ROOT/darcy_notch_l2/s0/logos_hetero --epochs 150 --batch 16 --freeze-frac 1.0 --seed 0 --noise feature --mask-domain
    $PY experiments/train_logos_vnngp.py --init $ROOT/navier_stokes_l2/s0/op --out $ROOT/navier_stokes_l2/s0/logos_hetero --epochs 100 --batch 16 --seed 0 --noise feature
    $PY experiments/train_logos_vnngp.py --init $ROOT/darcy/s0/op --out $ROOT/darcy/s0/logos_hetero --epochs 100 --batch 16 --seed 0 --noise feature --noise-head mlp --ckpt-every 5 --select nll
    $PY experiments/train_logos_vnngp.py --init $ROOT/airfoil/s0/op --out $ROOT/airfoil/s0/logos_hetero --epochs 100 --batch 16 --seed 0 --noise feature --ckpt-every 5 --select nll
}

sweeps() {   # grid-size (grid_sub<k>) and training-size (n<N>) sweeps of the scaling figures
    # advection grid_sub1
    $PY experiments/train_operator.py --data advection_paper --ntrain 2000 --ntest 100 --modes 32 --epochs 100 --seed 0 --out $ROOT/sweeps/advection/grid_sub1/op --arch ewno --level 3 --h1 0.1 --aug shift
    $PY experiments/train_logos_vnngp.py --init $ROOT/sweeps/advection/grid_sub1/op --out $ROOT/sweeps/advection/grid_sub1/logos --epochs 50 --batch 32 --seed 0
    $PY experiments/train_gpo.py --init $ROOT/sweeps/advection/grid_sub1/op --out $ROOT/sweeps/advection/grid_sub1/gpo --steps 100 --seed 0
    # advection grid_sub2
    $PY experiments/train_operator.py --data advection_paper@sub=2 --ntrain 2000 --ntest 100 --modes 32 --epochs 100 --seed 0 --out $ROOT/sweeps/advection/grid_sub2/op --arch ewno --level 3 --h1 0.1 --aug shift
    $PY experiments/train_logos_vnngp.py --init $ROOT/sweeps/advection/grid_sub2/op --out $ROOT/sweeps/advection/grid_sub2/logos --epochs 50 --batch 32 --seed 0
    $PY experiments/train_gpo.py --init $ROOT/sweeps/advection/grid_sub2/op --out $ROOT/sweeps/advection/grid_sub2/gpo --steps 100 --seed 0
    # advection grid_sub4
    $PY experiments/train_operator.py --data advection_paper@sub=4 --ntrain 2000 --ntest 100 --modes 25 --epochs 100 --seed 0 --out $ROOT/sweeps/advection/grid_sub4/op --arch ewno --level 2 --h1 0.1 --aug shift
    $PY experiments/train_logos_vnngp.py --init $ROOT/sweeps/advection/grid_sub4/op --out $ROOT/sweeps/advection/grid_sub4/logos --epochs 50 --batch 32 --seed 0
    $PY experiments/train_gpo.py --init $ROOT/sweeps/advection/grid_sub4/op --out $ROOT/sweeps/advection/grid_sub4/gpo --steps 100 --seed 0
    # advection grid_sub8
    $PY experiments/train_operator.py --data advection_paper@sub=8 --ntrain 2000 --ntest 100 --modes 12 --epochs 100 --seed 0 --out $ROOT/sweeps/advection/grid_sub8/op --arch ewno --level 1 --h1 0.1 --aug shift
    $PY experiments/train_logos_vnngp.py --init $ROOT/sweeps/advection/grid_sub8/op --out $ROOT/sweeps/advection/grid_sub8/logos --epochs 50 --batch 32 --seed 0
    $PY experiments/train_gpo.py --init $ROOT/sweeps/advection/grid_sub8/op --out $ROOT/sweeps/advection/grid_sub8/gpo --steps 100 --seed 0
    # advection n100
    $PY experiments/train_operator.py --data advection_paper --ntrain 100 --ntest 100 --modes 32 --epochs 100 --seed 0 --out $ROOT/sweeps/advection/n100/op --arch ewno --level 3 --h1 0.1 --aug shift
    $PY experiments/train_logos_vnngp.py --init $ROOT/sweeps/advection/n100/op --out $ROOT/sweeps/advection/n100/logos --epochs 750 --batch 32 --inducing 100 --seed 0
    $PY experiments/train_gpo.py --init $ROOT/sweeps/advection/n100/op --out $ROOT/sweeps/advection/n100/gpo --steps 100 --seed 0
    # advection n1000
    $PY experiments/train_operator.py --data advection_paper --ntrain 1000 --ntest 100 --modes 32 --epochs 100 --seed 0 --out $ROOT/sweeps/advection/n1000/op --arch ewno --level 3 --h1 0.1 --aug shift
    $PY experiments/train_logos_vnngp.py --init $ROOT/sweeps/advection/n1000/op --out $ROOT/sweeps/advection/n1000/logos --epochs 94 --batch 32 --seed 0
    $PY experiments/train_gpo.py --init $ROOT/sweeps/advection/n1000/op --out $ROOT/sweeps/advection/n1000/gpo --steps 100 --seed 0
    # advection n2000
    $PY experiments/train_operator.py --data advection_paper --ntrain 2000 --ntest 100 --modes 32 --epochs 100 --seed 0 --out $ROOT/sweeps/advection/n2000/op --arch ewno --level 3 --h1 0.1 --aug shift
    $PY experiments/train_logos_vnngp.py --init $ROOT/sweeps/advection/n2000/op --out $ROOT/sweeps/advection/n2000/logos --epochs 50 --batch 32 --seed 0
    $PY experiments/train_gpo.py --init $ROOT/sweeps/advection/n2000/op --out $ROOT/sweeps/advection/n2000/gpo --steps 100 --seed 0
    # advection n2700
    $PY experiments/train_operator.py --data advection_paper --ntrain 2700 --ntest 100 --modes 32 --epochs 100 --seed 0 --out $ROOT/sweeps/advection/n2700/op --arch ewno --level 3 --h1 0.1 --aug shift
    $PY experiments/train_logos_vnngp.py --init $ROOT/sweeps/advection/n2700/op --out $ROOT/sweeps/advection/n2700/logos --epochs 50 --batch 32 --seed 0
    $PY experiments/train_gpo.py --init $ROOT/sweeps/advection/n2700/op --out $ROOT/sweeps/advection/n2700/gpo --steps 100 --seed 0
    # advection n500
    $PY experiments/train_operator.py --data advection_paper --ntrain 500 --ntest 100 --modes 32 --epochs 100 --seed 0 --out $ROOT/sweeps/advection/n500/op --arch ewno --level 3 --h1 0.1 --aug shift
    $PY experiments/train_logos_vnngp.py --init $ROOT/sweeps/advection/n500/op --out $ROOT/sweeps/advection/n500/logos --epochs 188 --batch 32 --seed 0
    $PY experiments/train_gpo.py --init $ROOT/sweeps/advection/n500/op --out $ROOT/sweeps/advection/n500/gpo --steps 100 --seed 0
    # burgers grid_sub128
    $PY experiments/train_operator.py --data burgers@sub=128 --ntrain 1900 --nval 48 --ntest 100 --tail-test --modes 16 --epochs 300 --batch 20 --seed 0 --out $ROOT/sweeps/burgers/grid_sub128/op --arch wno --level 8 --wavelet db6 --h1 0.1 --aug shift
    $PY experiments/train_logos_vnngp.py --init $ROOT/sweeps/burgers/grid_sub128/op --out $ROOT/sweeps/burgers/grid_sub128/logos --epochs 100 --seed 0
    $PY experiments/train_gpo.py --init $ROOT/sweeps/burgers/grid_sub128/op --out $ROOT/sweeps/burgers/grid_sub128/gpo --steps 100 --seed 0
    # burgers grid_sub16
    $PY experiments/train_operator.py --data burgers@sub=16 --ntrain 1900 --nval 48 --ntest 100 --tail-test --modes 16 --epochs 300 --batch 20 --seed 0 --out $ROOT/sweeps/burgers/grid_sub16/op --arch wno --level 8 --wavelet db6 --h1 0.1 --aug shift
    $PY experiments/train_logos_vnngp.py --init $ROOT/sweeps/burgers/grid_sub16/op --out $ROOT/sweeps/burgers/grid_sub16/logos --epochs 100 --seed 0
    $PY experiments/train_gpo.py --init $ROOT/sweeps/burgers/grid_sub16/op --out $ROOT/sweeps/burgers/grid_sub16/gpo --steps 100 --seed 0
    # burgers grid_sub32
    $PY experiments/train_operator.py --data burgers@sub=32 --ntrain 1900 --nval 48 --ntest 100 --tail-test --modes 16 --epochs 300 --batch 20 --seed 0 --out $ROOT/sweeps/burgers/grid_sub32/op --arch wno --level 8 --wavelet db6 --h1 0.1 --aug shift
    $PY experiments/train_logos_vnngp.py --init $ROOT/sweeps/burgers/grid_sub32/op --out $ROOT/sweeps/burgers/grid_sub32/logos --epochs 100 --seed 0
    $PY experiments/train_gpo.py --init $ROOT/sweeps/burgers/grid_sub32/op --out $ROOT/sweeps/burgers/grid_sub32/gpo --steps 100 --seed 0
    # burgers grid_sub4
    $PY experiments/train_operator.py --data burgers@sub=4 --ntrain 1900 --nval 48 --ntest 100 --tail-test --modes 16 --epochs 300 --batch 20 --seed 0 --out $ROOT/sweeps/burgers/grid_sub4/op --arch wno --level 8 --wavelet db6 --h1 0.1 --aug shift
    $PY experiments/train_logos_vnngp.py --init $ROOT/sweeps/burgers/grid_sub4/op --out $ROOT/sweeps/burgers/grid_sub4/logos --epochs 100 --seed 0
    $PY experiments/train_gpo.py --init $ROOT/sweeps/burgers/grid_sub4/op --out $ROOT/sweeps/burgers/grid_sub4/gpo --steps 100 --seed 0
    # burgers grid_sub64
    $PY experiments/train_operator.py --data burgers@sub=64 --ntrain 1900 --nval 48 --ntest 100 --tail-test --modes 16 --epochs 300 --batch 20 --seed 0 --out $ROOT/sweeps/burgers/grid_sub64/op --arch wno --level 8 --wavelet db6 --h1 0.1 --aug shift
    $PY experiments/train_logos_vnngp.py --init $ROOT/sweeps/burgers/grid_sub64/op --out $ROOT/sweeps/burgers/grid_sub64/logos --epochs 100 --seed 0
    $PY experiments/train_gpo.py --init $ROOT/sweeps/burgers/grid_sub64/op --out $ROOT/sweeps/burgers/grid_sub64/gpo --steps 100 --seed 0
    # burgers grid_sub8
    $PY experiments/train_operator.py --data burgers --ntrain 1900 --nval 48 --ntest 100 --tail-test --modes 16 --epochs 300 --batch 20 --seed 0 --out $ROOT/sweeps/burgers/grid_sub8/op --arch wno --level 8 --wavelet db6 --h1 0.1 --aug shift
    $PY experiments/train_logos_vnngp.py --init $ROOT/sweeps/burgers/grid_sub8/op --out $ROOT/sweeps/burgers/grid_sub8/logos --epochs 100 --seed 0
    $PY experiments/train_gpo.py --init $ROOT/sweeps/burgers/grid_sub8/op --out $ROOT/sweeps/burgers/grid_sub8/gpo --steps 100 --seed 0
    # burgers n100
    $PY experiments/train_operator.py --data burgers@sub=16 --ntrain 100 --nval 48 --ntest 100 --tail-test --modes 16 --epochs 300 --batch 20 --seed 0 --out $ROOT/sweeps/burgers/n100/op --arch wno --level 8 --wavelet db6 --h1 0.1 --aug shift
    $PY experiments/train_logos_vnngp.py --init $ROOT/sweeps/burgers/n100/op --out $ROOT/sweeps/burgers/n100/logos --epochs 600 --inducing 100 --seed 0
    $PY experiments/train_gpo.py --init $ROOT/sweeps/burgers/n100/op --out $ROOT/sweeps/burgers/n100/gpo --steps 100 --seed 0
    # burgers n1000
    $PY experiments/train_operator.py --data burgers@sub=16 --ntrain 1000 --nval 48 --ntest 100 --tail-test --modes 16 --epochs 300 --batch 20 --seed 0 --out $ROOT/sweeps/burgers/n1000/op --arch wno --level 8 --wavelet db6 --h1 0.1 --aug shift
    $PY experiments/train_logos_vnngp.py --init $ROOT/sweeps/burgers/n1000/op --out $ROOT/sweeps/burgers/n1000/logos --epochs 100 --seed 0
    $PY experiments/train_gpo.py --init $ROOT/sweeps/burgers/n1000/op --out $ROOT/sweeps/burgers/n1000/gpo --steps 100 --seed 0
    # burgers n1900
    $PY experiments/train_operator.py --data burgers@sub=16 --ntrain 1900 --nval 48 --ntest 100 --tail-test --modes 16 --epochs 300 --batch 20 --seed 0 --out $ROOT/sweeps/burgers/n1900/op --arch wno --level 8 --wavelet db6 --h1 0.1 --aug shift
    $PY experiments/train_logos_vnngp.py --init $ROOT/sweeps/burgers/n1900/op --out $ROOT/sweeps/burgers/n1900/logos --epochs 100 --seed 0
    $PY experiments/train_gpo.py --init $ROOT/sweeps/burgers/n1900/op --out $ROOT/sweeps/burgers/n1900/gpo --steps 100 --seed 0
    # burgers n250
    $PY experiments/train_operator.py --data burgers@sub=16 --ntrain 250 --nval 48 --ntest 100 --tail-test --modes 16 --epochs 300 --batch 20 --seed 0 --out $ROOT/sweeps/burgers/n250/op --arch wno --level 8 --wavelet db6 --h1 0.1 --aug shift
    $PY experiments/train_logos_vnngp.py --init $ROOT/sweeps/burgers/n250/op --out $ROOT/sweeps/burgers/n250/logos --epochs 231 --inducing 250 --seed 0
    $PY experiments/train_gpo.py --init $ROOT/sweeps/burgers/n250/op --out $ROOT/sweeps/burgers/n250/gpo --steps 100 --seed 0
    # burgers n500
    $PY experiments/train_operator.py --data burgers@sub=16 --ntrain 500 --nval 48 --ntest 100 --tail-test --modes 16 --epochs 300 --batch 20 --seed 0 --out $ROOT/sweeps/burgers/n500/op --arch wno --level 8 --wavelet db6 --h1 0.1 --aug shift
    $PY experiments/train_logos_vnngp.py --init $ROOT/sweeps/burgers/n500/op --out $ROOT/sweeps/burgers/n500/logos --epochs 120 --seed 0
    $PY experiments/train_gpo.py --init $ROOT/sweeps/burgers/n500/op --out $ROOT/sweeps/burgers/n500/gpo --steps 100 --seed 0
    # notch grid_sub1
    $PY experiments/train_operator.py --data darcy_notch@sub=1 --ntrain 1500 --ntest 100 --epochs 100 --batch 16 --lr 0.0005 --seed 0 --out $ROOT/sweeps/notch/grid_sub1/op --arch wno --level 3 --wavelet db6 --h1 0.1
    $PY experiments/train_logos_vnngp.py --init $ROOT/sweeps/notch/grid_sub1/op --out $ROOT/sweeps/notch/grid_sub1/logos --epochs 50 --batch 16 --seed 0
    $PY experiments/train_gpo.py --init $ROOT/sweeps/notch/grid_sub1/op --out $ROOT/sweeps/notch/grid_sub1/gpo --steps 100 --seed 0
    # notch grid_sub2
    $PY experiments/train_operator.py --data darcy_notch --ntrain 1500 --ntest 100 --epochs 100 --batch 16 --lr 0.0005 --seed 0 --out $ROOT/sweeps/notch/grid_sub2/op --arch wno --level 2 --wavelet db6 --h1 0.1
    $PY experiments/train_logos_vnngp.py --init $ROOT/sweeps/notch/grid_sub2/op --out $ROOT/sweeps/notch/grid_sub2/logos --epochs 50 --batch 16 --seed 0
    $PY experiments/train_gpo.py --init $ROOT/sweeps/notch/grid_sub2/op --out $ROOT/sweeps/notch/grid_sub2/gpo --steps 100 --seed 0
    # notch grid_sub3
    $PY experiments/train_operator.py --data darcy_notch@sub=3 --ntrain 1500 --ntest 100 --epochs 100 --batch 16 --lr 0.0005 --seed 0 --out $ROOT/sweeps/notch/grid_sub3/op --arch wno --level 1 --wavelet db6 --h1 0.1
    $PY experiments/train_logos_vnngp.py --init $ROOT/sweeps/notch/grid_sub3/op --out $ROOT/sweeps/notch/grid_sub3/logos --epochs 50 --batch 16 --seed 0
    $PY experiments/train_gpo.py --init $ROOT/sweeps/notch/grid_sub3/op --out $ROOT/sweeps/notch/grid_sub3/gpo --steps 100 --seed 0
    # notch grid_sub4
    $PY experiments/train_operator.py --data darcy_notch@sub=4 --ntrain 1500 --ntest 100 --modes 13 13 --epochs 100 --batch 16 --lr 0.0005 --seed 0 --out $ROOT/sweeps/notch/grid_sub4/op --arch wno --level 1 --wavelet db6 --h1 0.1
    $PY experiments/train_logos_vnngp.py --init $ROOT/sweeps/notch/grid_sub4/op --out $ROOT/sweeps/notch/grid_sub4/logos --epochs 50 --batch 16 --seed 0
    $PY experiments/train_gpo.py --init $ROOT/sweeps/notch/grid_sub4/op --out $ROOT/sweeps/notch/grid_sub4/gpo --steps 100 --seed 0
    # notch n100
    $PY experiments/train_operator.py --data darcy_notch --ntrain 100 --ntest 100 --epochs 100 --batch 16 --lr 0.0005 --seed 0 --out $ROOT/sweeps/notch/n100/op --arch wno --level 2 --wavelet db6 --h1 0.1
    $PY experiments/train_logos_vnngp.py --init $ROOT/sweeps/notch/n100/op --out $ROOT/sweeps/notch/n100/logos --epochs 429 --batch 16 --inducing 100 --seed 0
    $PY experiments/train_gpo.py --init $ROOT/sweeps/notch/n100/op --out $ROOT/sweeps/notch/n100/gpo --steps 100 --seed 0
    # notch n1000
    $PY experiments/train_operator.py --data darcy_notch --ntrain 1000 --ntest 100 --epochs 100 --batch 16 --lr 0.0005 --seed 0 --out $ROOT/sweeps/notch/n1000/op --arch wno --level 2 --wavelet db6 --h1 0.1
    $PY experiments/train_logos_vnngp.py --init $ROOT/sweeps/notch/n1000/op --out $ROOT/sweeps/notch/n1000/logos --epochs 50 --batch 16 --seed 0
    $PY experiments/train_gpo.py --init $ROOT/sweeps/notch/n1000/op --out $ROOT/sweeps/notch/n1000/gpo --steps 100 --seed 0
    # notch n1500
    $PY experiments/train_operator.py --data darcy_notch --ntrain 1500 --ntest 100 --epochs 100 --batch 16 --lr 0.0005 --seed 0 --out $ROOT/sweeps/notch/n1500/op --arch wno --level 2 --wavelet db6 --h1 0.1
    $PY experiments/train_logos_vnngp.py --init $ROOT/sweeps/notch/n1500/op --out $ROOT/sweeps/notch/n1500/logos --epochs 50 --batch 16 --seed 0
    $PY experiments/train_gpo.py --init $ROOT/sweeps/notch/n1500/op --out $ROOT/sweeps/notch/n1500/gpo --steps 100 --seed 0
    # notch n200
    $PY experiments/train_operator.py --data darcy_notch --ntrain 200 --ntest 100 --epochs 100 --batch 16 --lr 0.0005 --seed 0 --out $ROOT/sweeps/notch/n200/op --arch wno --level 2 --wavelet db6 --h1 0.1
    $PY experiments/train_logos_vnngp.py --init $ROOT/sweeps/notch/n200/op --out $ROOT/sweeps/notch/n200/logos --epochs 231 --batch 16 --inducing 200 --seed 0
    $PY experiments/train_gpo.py --init $ROOT/sweeps/notch/n200/op --out $ROOT/sweeps/notch/n200/gpo --steps 100 --seed 0
    # notch n500
    $PY experiments/train_operator.py --data darcy_notch --ntrain 500 --ntest 100 --epochs 100 --batch 16 --lr 0.0005 --seed 0 --out $ROOT/sweeps/notch/n500/op --arch wno --level 2 --wavelet db6 --h1 0.1
    $PY experiments/train_logos_vnngp.py --init $ROOT/sweeps/notch/n500/op --out $ROOT/sweeps/notch/n500/logos --epochs 94 --batch 16 --seed 0
    $PY experiments/train_gpo.py --init $ROOT/sweeps/notch/n500/op --out $ROOT/sweeps/notch/n500/gpo --steps 100 --seed 0
    # ns grid_sub1
    $PY experiments/train_operator.py --data ns_forcing --ntrain 3000 --ntest 100 --width 48 --epochs 100 --lr 0.0003 --seed 0 --out $ROOT/sweeps/ns/grid_sub1/op --arch wnof --level 2 --wavelet db6 --h1 0.1
    $PY experiments/train_logos_vnngp.py --init $ROOT/sweeps/ns/grid_sub1/op --out $ROOT/sweeps/ns/grid_sub1/logos --epochs 50 --batch 16 --seed 0
    $PY experiments/train_gpo.py --init $ROOT/sweeps/ns/grid_sub1/op --out $ROOT/sweeps/ns/grid_sub1/gpo --steps 100 --seed 0
    # ns grid_sub2
    $PY experiments/train_operator.py --data ns_forcing@sub=2 --ntrain 3000 --ntest 100 --width 48 --epochs 100 --lr 0.0003 --seed 0 --out $ROOT/sweeps/ns/grid_sub2/op --arch wnof --level 1 --wavelet db6 --h1 0.1
    $PY experiments/train_logos_vnngp.py --init $ROOT/sweeps/ns/grid_sub2/op --out $ROOT/sweeps/ns/grid_sub2/logos --epochs 50 --batch 16 --seed 0
    $PY experiments/train_gpo.py --init $ROOT/sweeps/ns/grid_sub2/op --out $ROOT/sweeps/ns/grid_sub2/gpo --steps 100 --seed 0
    # ns grid_sub4
    $PY experiments/train_operator.py --data ns_forcing@sub=4 --ntrain 3000 --ntest 100 --modes 8 8 --width 48 --epochs 100 --lr 0.0003 --seed 0 --out $ROOT/sweeps/ns/grid_sub4/op --arch wnof --level 1 --wavelet db6 --h1 0.1
    $PY experiments/train_logos_vnngp.py --init $ROOT/sweeps/ns/grid_sub4/op --out $ROOT/sweeps/ns/grid_sub4/logos --epochs 50 --batch 16 --seed 0
    $PY experiments/train_gpo.py --init $ROOT/sweeps/ns/grid_sub4/op --out $ROOT/sweeps/ns/grid_sub4/gpo --steps 100 --seed 0
    # ns n100
    $PY experiments/train_operator.py --data ns_forcing --ntrain 100 --ntest 100 --width 48 --epochs 100 --lr 0.0003 --seed 0 --out $ROOT/sweeps/ns/n100/op --arch wnof --level 2 --wavelet db6 --h1 0.1
    $PY experiments/train_logos_vnngp.py --init $ROOT/sweeps/ns/n100/op --out $ROOT/sweeps/ns/n100/logos --epochs 429 --batch 16 --inducing 100 --seed 0
    $PY experiments/train_gpo.py --init $ROOT/sweeps/ns/n100/op --out $ROOT/sweeps/ns/n100/gpo --steps 100 --seed 0
    # ns n1000
    $PY experiments/train_operator.py --data ns_forcing --ntrain 1000 --ntest 100 --width 48 --epochs 100 --lr 0.0003 --seed 0 --out $ROOT/sweeps/ns/n1000/op --arch wnof --level 2 --wavelet db6 --h1 0.1
    $PY experiments/train_logos_vnngp.py --init $ROOT/sweeps/ns/n1000/op --out $ROOT/sweeps/ns/n1000/logos --epochs 50 --batch 16 --seed 0
    $PY experiments/train_gpo.py --init $ROOT/sweeps/ns/n1000/op --out $ROOT/sweeps/ns/n1000/gpo --steps 100 --seed 0
    # ns n2000
    $PY experiments/train_operator.py --data ns_forcing --ntrain 2000 --ntest 100 --width 48 --epochs 100 --lr 0.0003 --seed 0 --out $ROOT/sweeps/ns/n2000/op --arch wnof --level 2 --wavelet db6 --h1 0.1
    $PY experiments/train_logos_vnngp.py --init $ROOT/sweeps/ns/n2000/op --out $ROOT/sweeps/ns/n2000/logos --epochs 50 --batch 16 --seed 0
    $PY experiments/train_gpo.py --init $ROOT/sweeps/ns/n2000/op --out $ROOT/sweeps/ns/n2000/gpo --steps 100 --seed 0
    # ns n500
    $PY experiments/train_operator.py --data ns_forcing --ntrain 500 --ntest 100 --width 48 --epochs 100 --lr 0.0003 --seed 0 --out $ROOT/sweeps/ns/n500/op --arch wnof --level 2 --wavelet db6 --h1 0.1
    $PY experiments/train_logos_vnngp.py --init $ROOT/sweeps/ns/n500/op --out $ROOT/sweeps/ns/n500/logos --epochs 94 --batch 16 --seed 0
    $PY experiments/train_gpo.py --init $ROOT/sweeps/ns/n500/op --out $ROOT/sweeps/ns/n500/gpo --steps 100 --seed 0
    # ns n5000
    $PY experiments/train_operator.py --data ns_forcing --ntrain 5000 --ntest 100 --width 48 --epochs 100 --lr 0.0003 --seed 0 --out $ROOT/sweeps/ns/n5000/op --arch wnof --level 2 --wavelet db6 --h1 0.1
    $PY experiments/train_logos_vnngp.py --init $ROOT/sweeps/ns/n5000/op --out $ROOT/sweeps/ns/n5000/logos --epochs 50 --batch 16 --seed 0
    $PY experiments/train_gpo.py --init $ROOT/sweeps/ns/n5000/op --out $ROOT/sweeps/ns/n5000/gpo --steps 100 --seed 0
}

timing() {   # optional: re-time every sweep point alone on a GPU (3 LoGoS-GPO epochs, 10 GPO steps);
             # paper_figs.py prefers these timings when present
    for d in "$ROOT"/sweeps/*/*; do
        case "$d" in */burgers/*) bs=20;; */advection/*) bs=32;; *) bs=16;; esac
        $PY experiments/train_logos_vnngp.py --init "$d/op" --epochs 3 --batch $bs --out "$d/timing/logos"
        $PY experiments/train_gpo.py --init "$d/op" --steps 10 --out "$d/timing/gpo"
    done
}

apps() {   # applications: CO2 storage with the pre-trained U-FNO as the mean, Navier-Stokes 1024^2
    # CO2: needs a U-FNO clone and its pre-trained checkpoint (see experiments/ufno_mean.py), e.g.
    #   UFNO_DIR=/path/to/ufno UFNO_CKPT=/path/to/sg_UFNO_4500_trained.pt bash scripts/reproduce.sh apps
    $PY experiments/ufno_mean.py "$UFNO_CKPT" --out $ROOT/co2_ufno/op --test-pred $ROOT/co2_ufno/op/test_pred.pt
    $PY experiments/co2_eval.py --pred $ROOT/co2_ufno/op/test_pred.pt --out $ROOT/co2_ufno/op/co2_metrics.json
    for s in 0 1 2; do
        $PY experiments/train_logos_vnngp.py --init $ROOT/co2_ufno/op --epochs 15 --batch 8 --lr 0.001 --lr-op 3e-05 \
            --freeze-frac 1.0 --noise feature --noise-head mlp --noise-init residual --select nll --ckpt-every 1 \
            --seed $s --save-pred --out $ROOT/co2_ufno/s$s/logos
        $PY experiments/co2_eval.py --pred $ROOT/co2_ufno/s$s/logos/test_pred.pt --out $ROOT/co2_ufno/s$s/logos/co2_metrics.json
    done
    # Navier-Stokes 1024^2: operator on every 8th grid point with exact symmetry augmentation, GP on the full grid
    $PY experiments/train_operator.py --data ns1024@sub=8 --ntrain 900 --nval 100 --ntest 200 --tail-test --arch fno \
        --modes 32 32 --width 64 --layers 4 --epochs 1000 --batch 16 --lr 0.001 --wd 0.0001 --pad 0 --h1 0.1 \
        --aug shift2d_d4p --ckpt-every 25 --seed 0 --out $ROOT/ns1024/op
    $PY experiments/train_logos_vnngp.py --init $ROOT/ns1024/op --data ns1024 --split 900 100 200 --resample 8 \
        --epochs 20 --batch 4 --lr 0.001 --lr-op 3e-05 --freeze-frac 1.0 --noise feature --noise-init residual \
        --select nll --ckpt-every 1 --seed 0 --out $ROOT/ns1024/logos
    $PY experiments/ns1024_eval.py $ROOT/ns1024/logos --group
}

flops() {   # FLOPs study (no training), architectures of the stage-1 runs above
    $PY experiments/flops.py --bench burgers --init $ROOT/burgers/s0/op --out $ROOT/flops
    $PY experiments/flops.py --bench advection --init $ROOT/advection/s1/op --out $ROOT/flops
    $PY experiments/flops.py --bench notch --init $ROOT/darcy_notch/s0/op --out $ROOT/flops
    $PY experiments/flops.py --bench ns --init $ROOT/navier_stokes/s0/op --out $ROOT/flops
}

case "$WHAT" in
    table|extra|hetero|sweeps|timing|flops|apps) $WHAT ;;
    all) table; extra; hetero; sweeps; flops ;;
    *) echo "usage: $0 [table|extra|hetero|sweeps|timing|flops|apps|all]"; exit 1 ;;
esac
