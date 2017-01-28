package pipelines

import java.io.PrintWriter

import breeze.linalg.{DenseVector, SparseVector}
import evaluation.BinaryClassifierEvaluator
import loaders.{AmazonReviewsDataLoader, LabeledData}
import nodes.learning.LogisticRegressionEstimator
import nodes.nlp._
import nodes.stats.TermFrequency
import nodes.util.CommonSparseFeatures
import org.apache.spark.{SparkConf, SparkContext}
import pipelines.Logging
import scopt.OptionParser
import workflow.{Pipeline, PipelineDatum}
import nodes.learning.incremental.Model
import nodes.learning.incremental.ModelAvgIncrementaLREstimator
import pipelines.AmazonAvgPipeline.AmazonReviewsConfig

import scala.collection.mutable.ListBuffer
import scala.util.Random

class IncrementalPipeline(trainData: LabeledData[Int, String],
                          conf: AmazonReviewsConfig,
                          numClasses: Int = 2,
                          stepSize: Double,
                          miniBatchFraction: Double,
                          regParam: Double = 1e-3,
                          oldModel: Option[PipelineDatum[Model]] = None) {
  val training = trainData.data
  val labels = trainData.labels


  private lazy val featurizer = Trim andThen
    LowerCase() andThen
    Tokenizer() andThen
    NGramsFeaturizer(1 to conf.nGrams) andThen
    TermFrequency(x => 1) andThen
    (CommonSparseFeatures[Seq[String]](conf.commonFeatures), training)

  private lazy val incrementalLREstimator = new ModelAvgIncrementaLREstimator[SparseVector[Double]](
    numClasses,
    stepSize,
    miniBatchFraction,
    regParam,
    conf.numIters,
    numFeatures = conf.commonFeatures
  )

  private lazy val featurizedData = featurizer.toPipeline(training)
  private val prevModel: Model = oldModel match {
    case Some(datum) => datum.get()
    case None => incrementalLREstimator.initialModel()
  }

  val (incLR, model) = incrementalLREstimator.withData(featurizedData.get(), labels, prevModel)
  val predictor = featurizer andThen incLR
}

object AmazonAvgPipeline extends Logging {
  val appName = "AmazonAvgPipeline"

  def run(sc: SparkContext, conf: AmazonReviewsConfig): Unit = {
    logInfo("Loading amazon reviews data.")
    val percentTrainData = 0.8
    val amazonData = AmazonReviewsDataLoader(sc, conf.dataLocation, conf.threshold).labeledData
    val data = amazonData.repartition(conf.numParts).cache()

    logInfo("Performing train-test split: " + (percentTrainData * 100).toString + "% train data.")
    val Array(trainDataRDD, testDataRDD) = data.randomSplit(Array(percentTrainData, 1-percentTrainData))
    val testData = LabeledData(testDataRDD)

    // incremental parameters:
    val numBatches = 100
    logInfo("Converting training data into " + numBatches.toString + " baches.")
    val zippedBatches = trainDataRDD.zipWithIndex()
    val allBatches = (0 until numBatches)
      .map(i => zippedBatches.filter(_._2.toInt % numBatches == i).map(_._1))
      .map(rdd => LabeledData(rdd))
    logInfo("Converted training data.")

    logInfo("Shuffling and setting up data batches.")
    val batches = Random.shuffle(allBatches)

    // Pipeline parameters
    val numClasses = 2
    val stepSize = 1.0
    val miniFrac = 0.10
    val regParam = 1e-3

    // record train and test error as we train on more and more data
    var trainErr = new ListBuffer[Double]()
    var testErr = new ListBuffer[Double]()
    var trainTimes = new ListBuffer[Double]()

    // I don't want to do this a lot
    var evaluationTestLabels = testData.labels.map(_ > 0)

    var prevModel: Option[PipelineDatum[Model]] = None

    var j = 1
    val maxLen = batches.length
    for (batch <- batches) {
      logInfo("Training incremental batch: (" + j + " / " + maxLen + ")")
      
      val startTime = System.currentTimeMillis()

      // Train this pipeline
      val incPipeline = new IncrementalPipeline(
        batch,
        conf,
        numBatches,
        stepSize,
        miniFrac,
        regParam,
        oldModel = prevModel
      )

      // Test this iteration of the pipeline
      var trainPredictions = incPipeline.predictor(batch.data)

      val endTime = System.currentTimeMillis()
      trainTimes += (endTime - startTime)

      var testPredictions = incPipeline.predictor(testData.data)

      var trainEval = BinaryClassifierEvaluator(trainPredictions.get.map(_ > 0), batch.labels.map(_ > 0))
      var testEval = BinaryClassifierEvaluator(testPredictions.get.map(_ > 0), evaluationTestLabels)

      // Record the error for later analysis
      trainErr += trainEval.error
      testErr += testEval.error

      logInfo("Test error: " + testEval.error)

      // Store the previous model for the next iteration
      prevModel = Some(incPipeline.model)
    }

    val finalIncrementalTest = testErr.last

    val incrementalTrainStr = trainErr.mkString(",")
    val incrementalTestStr = testErr.mkString(",")
    val incrementalTimesStr = trainTimes.mkString(",")

    trainErr = new ListBuffer[Double]()
    testErr = new ListBuffer[Double]()
    trainTimes = new ListBuffer[Double]()

    logInfo("Performing baseline test.")
    var unionedBatches = batches.head.labeledData
    j = 1
    for (currentBatch <- batches) {
      logInfo("Training baseline batch: (" + j + " / " + maxLen + ")")
      unionedBatches = unionedBatches.union(currentBatch.labeledData)
      var labeledBatch = LabeledData(unionedBatches)

      val startTime = System.currentTimeMillis()

      // Train this pipeline
      val incPipeline = new IncrementalPipeline(
        labeledBatch,
        conf,
        numBatches,
        stepSize,
        miniFrac,
        regParam,
        oldModel = prevModel
      )

      // Test this iteration of the pipeline
      var trainPredictions = incPipeline.predictor(labeledBatch.data)
      
      val endTime = System.currentTimeMillis()
      trainTimes += (endTime - startTime)

      var testPredictions = incPipeline.predictor(testData.data)

      var trainEval = BinaryClassifierEvaluator(trainPredictions.get.map(_ > 0), labeledBatch.labels.map(_ > 0))
      var testEval = BinaryClassifierEvaluator(testPredictions.get.map(_ > 0), evaluationTestLabels)

      // Record the error for later analysis
      trainErr += trainEval.error
      testErr += testEval.error
      trainTimes += (endTime - startTime)

      logInfo("Test error: " + testEval.error)

      // Store the previous model for the next iteration
      prevModel = Some(incPipeline.model)
    }

    val finalBaselineTest = testErr.last

    val baselineTrainStr = trainErr.mkString(",")
    val baselineTestStr = testErr.mkString(",")
    val baselineTimesStr = trainTimes.mkString(",")

    logInfo("Writing out train and test scores...")

    val trainStr = incrementalTrainStr + "\n" + baselineTrainStr
    val testStr = incrementalTestStr + "\n" + baselineTestStr
    val timesStr = incrementalTimesStr + "\n" + baselineTimesStr

    new PrintWriter("trainErrors.csv") { write(trainStr); close }
    new PrintWriter("testErrors.csv") { write(testStr); close }
    new PrintWriter("trainTimes.csv") { write(timesStr); close }

    logInfo("Finished.")
    logInfo("Final incremental test accuracy: " + finalIncrementalTest)
    logInfo("Final baseline test accuracy: " + finalBaselineTest)


  }

  case class AmazonReviewsConfig(
                                  dataLocation: String = "",
                                  threshold: Double = 3.5,
                                  nGrams: Int = 2,
                                  commonFeatures: Int = 100000,
                                  numIters: Int = 20,
                                  numParts: Int = 512)

  def parse(args: Array[String]): AmazonReviewsConfig = new OptionParser[AmazonReviewsConfig](appName) {
    head(appName, "0.1")
    opt[String]("dataLocation") required() action { (x,c) => c.copy(dataLocation=x) }
    opt[Double]("threshold") action { (x,c) => c.copy(threshold=x)}
    opt[Int]("nGrams") action { (x,c) => c.copy(nGrams=x) }
    opt[Int]("commonFeatures") action { (x,c) => c.copy(commonFeatures=x) }
    opt[Int]("numIters") action { (x,c) => c.copy(numParts=x) }
    opt[Int]("numParts") action { (x,c) => c.copy(numParts=x) }
  }.parse(args, AmazonReviewsConfig()).get

  /**
    * The actual driver receives its configuration parameters from spark-submit usually.
    *
    * @param args
    */
  def main(args: Array[String]) = {
    val conf = new SparkConf().setAppName(appName)
    conf.setIfMissing("spark.master", "local[2]") // This is a fallback if things aren't set via spark submit.

    val sc = new SparkContext(conf)

    val appConfig = parse(args)
    run(sc, appConfig)

    sc.stop()
  }
}
