package pipelines

import breeze.linalg.DenseVector
import loaders.LabeledData
import nodes.Word2VecLoader
import nodes.learning._
import org.apache.spark.mllib.classification.{LogisticRegressionWithSGD, LogisticRegressionModel => MLlibLRM}
import nodes.learning.incremental.{IncrementalLogisticRegressionEstimator, ModelAvgIncrementaLREstimator}
import nodes.util.{ClassLabelIndicatorsFromIntLabels, MaxClassifier, VectorCombiner}
import workflow.{PipelineDatum, _}


/**
  * Created by noah on 4/26/16.
  */
class IncrementalClassifierPipeline(labeledData: LabeledData[Int, String],
                                    numClasses: Int,
                                    stepSize: Double,
                                    miniBatchFraction: Double,
                                    previousPredictor: Option[Pipeline[String, DenseVector[Double]]] = None,
                                    numIterations: Int = 25,
                                    regParam: Double = 1e-3,
                                    oldModel: Option[PipelineDatum[MLlibLRM]] = None) {
  val labels = ClassLabelIndicatorsFromIntLabels(numClasses = numClasses)(labeledData.labels)

  private lazy val featurizer = {
    val w2v = new Word2VecLoader
    if (previousPredictor.isDefined) Pipeline.gather[String, DenseVector[Double]](Seq(w2v.toPipeline, previousPredictor.get)) andThen VectorCombiner[Double]
    else w2v
  }

  private lazy val incrementalLREstimator = new ModelAvgIncrementaLREstimator[DenseVector[Double]](
    numClasses,
    stepSize,
    miniBatchFraction,
    regParam,
    numIterations,
    numFeatures = Word2VecLoader.numDims
  )

  val featurizedData = featurizer.toPipeline(labeledData.data)

  val prevModel = oldModel match {
    case Some(datum) => datum
    case None => incrementalLREstimator.getInitialModel()
  }

  val (incLR, model) = incrementalLREstimator.withData(featurizedData, labeledData.labels, prevModel)

  lazy val predictor = featurizer andThen incLR

  lazy val classifier = predictor // andThen MaxClassifier

}
