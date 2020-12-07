from argparse import ArgumentParser
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession


MODEL_PATH = 'spark_ml_model'


parser = ArgumentParser()

parser.add_argument(
    '--test', default='test.parquet', type=str, required=False,
    help='Path to train data source'
)

parser.add_argument(
    '--result', default='result', type=str, required=False,
    help='Path to result data'
)

args = parser.parse_args()


def process(spark, input_file, output_file, partitions=1):
    df = spark.read.parquet(input_file)
    loaded_model = PipelineModel.load(MODEL_PATH)
    prediction = loaded_model.transform(df)
    prediction.select('ad_id', 'prediction').coalesce(partitions).write.parquet(output_file)
    spark.stop()


def main(argv):
    input_path = args.test
    print("Input path to file: " + input_path)
    output_file = args.result
    print("Output path to file: " + output_file)
    spark = _spark_session()
    process(spark, input_path, output_file)


def _spark_session():
    return SparkSession.builder.appName('PySparkMLPredict').getOrCreate()


if __name__ == "__main__":
    main(args)
