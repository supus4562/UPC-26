"""Nine-zone projection tables with empirical predictive uncertainty."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from data import load_partition

METHODS = ('t1', 't2', 'both', 'control')
LABELS = [f'{v} {h}' for v in ('low','middle','high') for h in ('left','center','right')]


def cell(q):
    q = np.asarray(q)
    return np.searchsorted([2.44/3, 4.88/3], q[...,1], side='right')*3 + np.searchsorted([-1.22,1.22], q[...,0], side='right')


def zones(x):
    p = np.asarray(x).reshape(-1,2,3)
    if np.any(p[:,:,1] <= 0):
        raise ValueError('Observed forward distances must be positive')
    projected = np.stack((11*p[:,:,0]/p[:,:,1], .11+11*(p[:,:,2]-.11)/p[:,:,1]), axis=-1)
    return cell(projected)


def keys(z, method):
    return {'t1':z[:,0], 't2':z[:,1], 'both':9*z[:,0]+z[:,1], 'control':np.zeros(len(z), dtype=int)}[method]


def summarize(q, weights):
    weights = weights/weights.sum()
    mean = weights@q
    residual = q-mean
    cov = (residual.T*weights)@residual
    # Tiny numerical floor defines ellipses for degenerate training groups.
    cov += np.eye(2)*1e-10
    distances = np.einsum('ni,ij,nj->n', residual, np.linalg.inv(cov), residual)
    order = np.argsort(distances)
    cutoff = float(distances[order[np.searchsorted(np.cumsum(weights[order]),.95)]])
    probabilities = np.bincount(cell(q), weights=weights, minlength=9)
    order = np.argsort(-probabilities, kind='stable')
    selected = order[:np.searchsorted(np.cumsum(probabilities[order]),.95)+1]
    return dict(mean=mean.tolist(), covariance=cov.tolist(), sigma=np.sqrt(np.diag(cov)).tolist(), radial_rms=float(np.sqrt(np.trace(cov))), probabilities=probabilities.tolist(), predicted_cell=int(probabilities.argmax()), cells95=selected.tolist(), ellipse_cutoff=cutoff, ellipse_area=float(np.pi*cutoff*np.sqrt(np.linalg.det(cov))))


def train(args):
    x,q,ids,meta = load_partition(args.dataset,'train')
    z=zones(x)
    model=dict(version=1, split=meta, train_ids=ids.tolist(), smoothing=args.smoothing, labels=LABELS, tables={})
    global_weights=np.ones(len(q))/len(q)
    for method, size in zip(METHODS,(9,9,81,1)):
        table=[]
        index=keys(z,method)
        for key in range(size):
            mask=index==key
            count=int(mask.sum())
            parent=global_weights
            if method=='both':
                parent_mask=z[:,1]==key%9
                if parent_mask.any():
                    parent=parent_mask/parent_mask.sum()
            weights=(mask.astype(float)+args.smoothing*parent)/(count+args.smoothing) if count else parent
            row=summarize(q,weights)
            row.update(count=count, fallback=count==0)
            table.append(row)
        model['tables'][method]=table
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    (output/'model.json').write_text(json.dumps(model))
    for method,table in model['tables'].items():
        with (output/f'{method}.csv').open('w',newline='') as stream:
            writer=csv.writer(stream)
            writer.writerow(['input','count','predicted_cell','mean_lateral','mean_height','sigma_lateral','sigma_height','radial_rms']+LABELS)
            for k,row in enumerate(table):
                label = f'{LABELS[k//9]} -> {LABELS[k%9]}' if method=='both' else ('all' if method=='control' else LABELS[k])
                writer.writerow([label,row['count'],LABELS[row['predicted_cell']],*row['mean'],*row['sigma'],row['radial_rms'],*row['probabilities']])
    print(f'Saved four tables and model in {output}')


def predictions(model,x,method):
    return [model['tables'][method][int(k)] for k in keys(zones(x),method)]


def evaluate(args):
    model=json.loads(Path(args.model).read_text())
    x,q,ids,meta=load_partition(args.dataset,'test')
    if meta != model['split'] or np.intersect1d(ids,model['train_ids']).size:
        raise ValueError('Split mismatch or training/test ID overlap')
    if args.limit: x,q=x[:args.limit],q[:args.limit]
    truth=cell(q)
    results={}
    noise=np.random.default_rng(args.seed).normal(0,args.sigma,x.shape)
    for scenario,observations in [('clean',x),('noisy',x+noise)]:
        results[scenario]={}
        for method in METHODS:
            rows=predictions(model,observations,method)
            mean=np.array([r['mean'] for r in rows]); prob=np.array([r['probabilities'] for r in rows])
            pred=cell(mean) if method=='control' else prob.argmax(axis=1)
            error=q-mean; mse=np.mean(error**2,axis=0)
            confusion=np.zeros((9,9),int); np.add.at(confusion,(truth,pred),1)
            cover=[]
            for e,r in zip(error,rows):
                cover.append(e@np.linalg.solve(r['covariance'],e)<=r['ellipse_cutoff']+1e-12)
            support=confusion.sum(axis=1); present=support>0
            results[scenario][method]=dict(n=len(q),accuracy=float(np.mean(pred==truth)),row_accuracy=float(np.mean(pred//3==truth//3)),column_accuracy=float(np.mean(pred%3==truth%3)),mean_grid_distance=float(np.mean(abs(pred//3-truth//3)+abs(pred%3-truth%3))),macro_recall=float(np.mean(np.diag(confusion)[present]/support[present])),confusion_matrix=confusion.tolist(),brier=float(np.mean(np.sum((prob-np.eye(9)[truth])**2,axis=1))),cell_set95_coverage=float(np.mean([int(t) in r['cells95'] for t,r in zip(truth,rows)])),cell_set95_mean_size=float(np.mean([len(r['cells95']) for r in rows])),mse_axes=mse.tolist(),radial_mse=float(mse.sum()),radial_rmse=float(np.sqrt(mse.sum())),mean_sigma_axes=np.mean([r['sigma'] for r in rows],axis=0).tolist(),mean_radial_rms=float(np.mean([r['radial_rms'] for r in rows])),ellipse95_coverage=float(np.mean(cover)),mean_ellipse_area=float(np.mean([r['ellipse_area'] for r in rows])),fallback_rate=float(np.mean([r['fallback'] for r in rows])))
    Path(args.output).write_text(json.dumps(dict(sigma=args.sigma,seed=args.seed,results=results),indent=2))
    print(f'Aggregate benchmarks saved to {args.output}')


def plot_prediction(result, path, frame):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse, Rectangle

    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5), layout='constrained')
    horizontal = 'y' if frame == 'problem' else 'x'
    maximum = max(max(row['probabilities']) for row in result['methods'].values())
    for ax, (method, row) in zip(axes.flat, result['methods'].items()):
        probabilities = np.asarray(row['probabilities']).reshape(3, 3)
        heatmap = ax.imshow(probabilities, origin='lower', extent=(-3.66, 3.66, 0, 2.44),
                            cmap='Blues', vmin=0, vmax=maximum, interpolation='nearest')
        for index, probability in enumerate(probabilities.ravel()):
            ax.text(-2.44 + 2.44*(index % 3), (index // 3 + .5)*2.44/3,
                    f'{probability:.1%}', ha='center', va='center',
                    color='white' if probability > .6*maximum else 'black')
        predicted = int(cell(row['mean'])) if method == 'control' else row['predicted_cell']
        ax.add_patch(Rectangle((-3.66 + 2.44*(predicted % 3), 2.44/3*(predicted // 3)),
                               2.44, 2.44/3, fill=False, edgecolor='#e17800',
                               linewidth=2.5, label='Predicted cell'))
        eigenvalues, eigenvectors = np.linalg.eigh(row['covariance'])
        angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
        diameters = 2*np.sqrt(np.maximum(eigenvalues, 0)*row['ellipse_cutoff'])
        ax.add_patch(Ellipse(row['mean'], diameters[1], diameters[0], angle=angle,
                             fill=False, edgecolor='#ad1645', linewidth=1.8,
                             label='Nominal 95% ellipse (clipped at frame)'))
        ax.plot(*row['mean'], marker='x', color='#ad1645', markersize=9,
                markeredgewidth=2, linestyle='none', label='Mean crossing')
        ax.set_xticks([-3.66, -1.22, 1.22, 3.66])
        ax.set_yticks([0, 2.44/3, 4.88/3, 2.44])
        ax.grid(alpha=.35)
        ax.set(xlim=(-3.66, 3.66), ylim=(0, 2.44), xlabel=f'{horizontal} (m)', ylabel='z (m)')
        ax.set_title(f"{method} | {LABELS[predicted]} | training n={row['count']}\n"
                     f"sigma=({row['sigma'][0]:.3f}, {row['sigma'][1]:.3f}) m; "
                     f"radial RMS={row['radial_rms']:.3f} m")
    fig.suptitle('Goal-cell probabilities | ' + ' → '.join(result['input_zones']))
    fig.colorbar(heatmap, ax=axes.ravel().tolist(), label='Probability per cell', shrink=.8)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncols=3)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def predict(args):
    model=json.loads(Path(args.model).read_text())
    points=np.array([args.r1,args.r2],float)
    if args.frame=='problem': points=points[:,[1,0,2]]
    z=zones(points.reshape(1,6))[0]
    result=dict(input_zones=[LABELS[i] for i in z],coordinate_order='lateral, height',methods={m:predictions(model,points.reshape(1,6),m)[0] for m in METHODS})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result,indent=2))
    png = args.png or str(Path(args.output).with_suffix('.png'))
    plot_prediction(result, png, args.frame)
    print(f'Saved PNG to {png}')
    for m,r in result['methods'].items():
        print(m, LABELS[r['predicted_cell']] if m!='control' else LABELS[int(cell(r['mean']))], 'mean=',r['mean'],'sigma=',r['sigma'])


def main(command=None):
    import sys
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('train'); p.add_argument('--dataset',required=True); p.add_argument('--output',default='grid_models'); p.add_argument('--smoothing',type=float,default=5)
    p=sub.add_parser('evaluate'); p.add_argument('--dataset',required=True); p.add_argument('--model',default='grid_models/model.json'); p.add_argument('--output',default='metrics_grid.json'); p.add_argument('--sigma',type=float,default=.005/np.sqrt(3)); p.add_argument('--seed',type=int,default=42); p.add_argument('--limit',type=int)
    p=sub.add_parser('predict'); p.add_argument('--model',default='grid_models/model.json'); p.add_argument('--output',default='prediction_grid.json'); p.add_argument('--r1',nargs=3,type=float,default=[1.50,.45,.44]); p.add_argument('--r2',nargs=3,type=float,default=[2.96,.86,.74]); p.add_argument('--frame',choices=['problem','paper'],default='problem')
    p.add_argument('--png', help='PNG path; defaults to the JSON output basename with .png')
    args=parser.parse_args(([command]+sys.argv[1:]) if command else None)
    if getattr(args,'smoothing',0)<0 or getattr(args,'sigma',0)<0: parser.error('Smoothing and sigma must be nonnegative')
    globals()[args.command](args)

if __name__=='__main__': main()
