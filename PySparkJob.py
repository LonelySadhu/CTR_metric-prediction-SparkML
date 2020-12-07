import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pathlib import Path


def process(spark, input_file, target_path):
    """Group input_file data, split the result to train/val/test and save them to folders inside target_path folder"""
    
    source = spark.read.parquet(input_file)
    
    grouped = source.groupby('ad_id').agg(
        f.first('target_audience_count').alias('target_audience_count'),
        f.first('has_video').alias('has_video'),
        (f.first('ad_cost_type') == 'CPM').astype('int').alias('is_cpm'),
        (f.first('ad_cost_type') == 'CPC').astype('int').alias('is_cpc'),
        f.first('ad_cost').alias('ad_cost'),
        f.countDistinct('date').alias('day_count'),
        (
            f.sum(f.when(f.col('event') == 'click', 1).otherwise(0)) /
            f.sum(f.when(f.col('event') == 'view', 1).otherwise(0))
        ).alias('ctr')
    ).filter('ctr is not null')
    
    train, test, validate = grouped.randomSplit([0.5, 0.25, 0.25], seed=42)
    
    target_folder = Path(target_path)
    train.coalesce(1).write.mode('overwrite').parquet(str(target_folder/'train'))
    test.coalesce(1).write.mode('overwrite').parquet(str(target_folder/'test'))
    validate.coalesce(1).write.mode('overwrite').parquet(str(target_folder/'validate'))


def main(argv):
    input_path = argv[0]
    print("Input path to file: " + input_path)
    target_path = argv[1]
    print("Target path: " + target_path)
    spark = _spark_session()
    process(spark, input_path, target_path)


def _spark_session():
    return SparkSession.builder.appName('PySparkJob').getOrCreate()


if __name__ == "__main__":
    arg = sys.argv[1:]
    if len(arg) != 2:
        sys.exit("Input and Target path are required.")
    else:
        main(arg)
