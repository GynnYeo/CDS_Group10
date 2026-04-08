# Hi, if you encounter any problems, look in here you might find a solution!

# 1. Unable to inspect the Processed Data (after you run build the dataset)

After you build your dataset, if you wish to inspect, run this first in your KERNEL as you need it to open .parquet files.

## Step 1: Uninstall previous versions of PyArrow

```bash
python pip uninstall pyarrow -y pip uninstall pandas -y
```

## Step 2: Reinstall the compatible versions

```bash
python pip install pandas==2.2.2 pyarrow==15.0.2
```
