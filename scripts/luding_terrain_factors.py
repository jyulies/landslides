# -*- coding: utf-8 -*-
"""
从泸定 GDB 里的 Elevation 派生 Slope / Aspect / Profile_Curvature / Plan_Curvature，
输出为 TIF 到 Dataset/luding/factor/（命名与项目流水线一致）。

DEM 已是 UTM 48N + 30m 米制像元，z_factor=1。
注意：用 gp.*_sa 直接写输出路径，而不是 Raster.save()——后者在中文
Windows 区域设置下读栅格元数据时可能抛 UnicodeDecodeError（GBK/UTF-8 冲突）。

用法（ArcGIS Pro 自带 Python）：
    "C:\\Program Files\\ArcGIS\\Pro\\bin\\Python\\envs\\arcgispro-py3\\python.exe" scripts/luding_terrain_factors.py
"""
import sys
from pathlib import Path

import arcpy

DEM = r"C:\Users\Jyuli\Documents\ArcGIS\Projects\luding\luding.gdb\Elevation"
OUT_DIR = Path(r"C:\code\landslides\Dataset\luding\factor")

arcpy.env.overwriteOutput = True
arcpy.CheckOutExtension("Spatial")


def describe_out(path, name):
    d = arcpy.Describe(path)
    print("✔ %-22s %dx%d  nodata=%s" % (name, d.width, d.height, d.noDataValue))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    desc = arcpy.Describe(DEM)
    assert desc.spatialReference.linearUnitName == "Meter", \
        "DEM 不是米制投影，需先投影或设置 z_factor！当前: " + desc.spatialReference.name

    out_slope = str(OUT_DIR / "Slope.tif")
    out_aspect = str(OUT_DIR / "Aspect.tif")
    out_prof = str(OUT_DIR / "Profile_Curvature.tif")
    out_plan = str(OUT_DIR / "Plan_Curvature.tif")

    # 坡度（度）；坡向（-1=平地，0~360）；曲率（总曲率输出不需要，写到内存占位）
    arcpy.gp.Slope_sa(DEM, out_slope, "DEGREE", 1)
    describe_out(out_slope, "Slope.tif")

    arcpy.gp.Aspect_sa(DEM, out_aspect, "GEODESIC", "METER")
    describe_out(out_aspect, "Aspect.tif")

    arcpy.gp.Curvature_sa(DEM, r"memory\curvature", 1, r"memory\prof_raw", r"memory\plan_raw")
    # ArcGIS Curvature 输出是真实曲率的 100 倍（单位 1/100 z 单位），
    # 除以 100 转成与其他区一致的 1/m（Times 不允许输出覆盖输入，故从内存写出）
    arcpy.gp.Times_sa(r"memory\prof_raw", 0.01, out_prof)
    arcpy.gp.Times_sa(r"memory\plan_raw", 0.01, out_plan)
    describe_out(out_prof, "Profile_Curvature.tif")
    describe_out(out_plan, "Plan_Curvature.tif")

    arcpy.CheckInExtension("Spatial")
    print("\n完成，输出目录:", OUT_DIR)


if __name__ == "__main__":
    sys.exit(main())
