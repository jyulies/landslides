import rasterio
from pathlib import Path

# sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

def main():
    with rasterio.open(Path("./factor/Elevation.tif")) as src:
        src_arr = src.read(1)
        src_nodata = src.nodata
        print("nodata:", src_nodata)
        print(Path(__file__).resolve().parents[2])


if __name__ == "__main__":
    main()



