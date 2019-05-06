package wi



import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.sql.{DataFrame, Row, SaveMode, SparkSession}
import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{MultilayerPerceptronClassificationModel, MultilayerPerceptronClassifier}
import org.apache.spark.ml.feature.{StopWordsRemover, Tokenizer, Word2Vec}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.types.DoubleType


object Main extends App {

  /**
    *Creates and safe the piple
    */
  def createPipeline(dataFrame: DataFrame,path : String): PipelineModel ={
    val tokenizer = new Tokenizer().setInputCol("comment_text").setOutputCol("words")
    val stopWord = new StopWordsRemover().setInputCol(tokenizer.getOutputCol).setOutputCol("stopWordFilter")
    val word2vec = new Word2Vec().setInputCol(stopWord.getOutputCol).setOutputCol("features")
    val PL = new Pipeline().setStages(Array(tokenizer, stopWord,word2vec)).fit(dataFrame)
    PL.write.overwrite().save(path)

    PL
  }
  /**
  Function to create and save the models
    **/
  def createModel(spark: SparkSession): MultilayerPerceptronClassificationModel =
  {
    val train = "Data/train.csv/train.csv"
    println("Loading data")
    val dfTrain = spark.read.option("header","true").option("inferSchema",true).csv(train)
    println("Preparing data")
    val dfCleaned = Cleaner.prepareDF(dfTrain) //Clean dataFrame

    val hadoopfs = FileSystem.get(spark.sparkContext.hadoopConfiguration)
    val PL = if(hadoopfs.exists(new Path("pipeline/PipeLineNN")))
      {
      PipelineModel.read.load("pipeline/PipeLineNN")
      }
      else
      {
        println("Creating the Pipeline")
         createPipeline(dfCleaned,"pipeline/PipeLineNN")
      }

    println("Creating the model")
    val data_train = PL.transform(dfCleaned).select("id",
      "features","target","male","female","transgender","other_gender",
      "heterosexual","homosexual_gay_or_lesbian","bisexual","other_sexual_orientation","christian","jewish","muslim",
      "hindu","buddhist","atheist","other_religion","black","white","asian","latino","other_race_or_ethnicity",
      "physical_disability","intellectual_or_learning_disability","psychiatric_or_mental_illness","other_disability").withColumn("target",dfCleaned("target").cast(DoubleType))

    val splits = data_train.randomSplit(Array(0.8,0.2),1234L)
    val data_split_train = splits(0)
    val data_split_test = splits(1)
    val layers = Array[Int](100,20, 20,2)
    val NN = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setSeed(1234L)
      .setMaxIter(1000)
      .setLabelCol("target")

    val nnModel = NN.fit(data_split_train)
    println("Model created!!! Saving.....")
    nnModel.write.overwrite().save("model/NNModel")
    println("model created ! Evaluation")
    val eval_data_train = nnModel.transform(data_split_test)
    val output = evaluation(eval_data_train,"target")
    println(output)

    nnModel
  }

  def useModel()= {
    val test = "Data/test.csv/test.csv"
    Logger.getLogger("org").setLevel(Level.ERROR) //Remove all the INFO prompt
    val spark = SparkSession.builder.appName("Comment Toxicity").config("spark.master", "local").getOrCreate() //Create spark session


    println("loading data")
    val dfTest = spark.read.option("header","true").option("inferSchema",true).csv(test)
    println("prapearin data")
    val dfCleaned = Cleaner.prepareDfForPrediction(dfTest)
    println("Loading the model and the pipeline")
    val hadoopfs = FileSystem.get(spark.sparkContext.hadoopConfiguration)
    val PL = if(hadoopfs.exists(new Path("pipeline/PipeLineNNPrediction")))
    {
      println("loading  pipeline")
      PipelineModel.read.load("pipeline/PipeLineNNPrediction")
    }
    else
    {
      println("Creating pipeline")
      createPipeline(dfCleaned,"pipeline/PipeLineNNPrediction")
    }

    val nnModel =  if(hadoopfs.exists(new Path("model/NNModel")))
      {
        println("loading Model")
        MultilayerPerceptronClassificationModel.load("model/NNModel")
      }
    else
      {
        println("Creating Model")
        createModel(spark)
      }

    val dfTestPrediction =  PL.transform(dfCleaned).select("id","features")

    val finalDf = nnModel.transform(dfCleaned)

    val result = finalDf.select("id","prediction")

    println("saving the results as a CSV")

    result.write.format("csv").option("header","true").mode(SaveMode.Overwrite).save("prediction/tmp/ResultRF")

    println("CSV available in prediction/tmp/ResultsRF")
    spark.stop()
  }

  // Description: Evaluate your model with various metrics
  // Parameter:
  // - data: DataFrame we need to evaluate the model on
  // - label_name: Name of Label Column
  // Return: string containing the result
  def evaluation(data: DataFrame,label_name: String): String = {
    // Select prediction and label column

    val pred_label = data.select("prediction","target").withColumnRenamed(label_name, "label")

    // Convert to RDD
    val eval_rdd = pred_label.rdd.map{case Row(prediction:Double,label:Double) =>(prediction,label)}

    val metric = new MulticlassMetrics(eval_rdd)

    val accuracy = metric.accuracy
    val fMeasure = metric.weightedFMeasure
    val precision = metric.weightedPrecision
    val recall = metric.weightedRecall
    val TPR = metric.weightedTruePositiveRate
    val FPR = metric.weightedFalsePositiveRate
    val result = "accuracy = "+ accuracy+"\nPrecision = "+precision+"\nRecall = "+recall+"\nFl = "+fMeasure+"\nTPR = "+TPR+
      "\nFRP = "+FPR

     result
  }



  useModel()

}
