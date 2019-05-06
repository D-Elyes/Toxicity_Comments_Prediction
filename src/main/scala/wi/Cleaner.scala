package wi

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.udf

object Cleaner {

  def cleanNullValues(d:DataFrame,c:Array[String]): DataFrame = {

    val newDf = d.na.drop(c)

    newDf
  }

  def cleanTarget(d:DataFrame):DataFrame = {
    val newDf = d.filter(d.col("target").between(0.0,1.0))

    newDf
  }


  def cleanRegularWords(c : String) : String = {
    var toClean = c

    toClean = toClean.replaceAll("\\baren't\\b|\\barent\\b","are not")
    toClean = toClean.replaceAll("\\bain't\\b","am not")
    toClean = toClean.replaceAll("\\bcan't\\b|\\bcannot\\b|\\bcant\\b","cannot")
    toClean = toClean.replaceAll("\\bcould've\\b","could have")
    toClean = toClean.replaceAll("\\bcouldn't\\b|\\bcouldnt\\b","could not")
    toClean = toClean.replaceAll("\\bisn't\\b|\\bisnt\\b","is not")
    toClean = toClean.replaceAll("\\bwasn't\\b|\\bwasnt\\b","was not")
    toClean = toClean.replaceAll("\\bweren't\\b|\\bwerent\\b","were not")
    toClean = toClean.replaceAll("\\bmustn't\\b|mustnt\\b","must not")
    toClean = toClean.replaceAll("\\bdidn't\\b|\\bdidnt\\b","did not")
    toClean = toClean.replaceAll("\\bdoesn't\\b|\\bdoesnt\\b","does not")
    toClean = toClean.replaceAll("\\bdon't\\b|\\bdont\\b","do not")
    toClean = toClean.replaceAll("\\bneedn't\\b|\\bneednt\\b","need not")
    toClean = toClean.replaceAll("\\bshouldn't\\b|\\bshouldnt\\b","i am")
    toClean = toClean.replaceAll("\\bhadn't\\b|\\bhadnt\\b","had not")
    toClean = toClean.replaceAll("\\bhasn't\\b|\\bhasnt\\b","has not")
    toClean = toClean.replaceAll("\\bhaven't\\b|havent\\b","have not")
    toClean = toClean.replaceAll("\\bi've\\b|\\bive\\b|\\bi ve\\b","i have")
    toClean = toClean.replaceAll("\\bwe've\\b|weve\\b|\\bwe ve\\b","we have")
    toClean = toClean.replaceAll("\\bthey've\\b|\\bthey ve\\b|\\btheyve\\b","they have")
    toClean = toClean.replaceAll("\\byou've\\b|\\byouve\\b|\\byou ve\\b","you have")
    toClean = toClean.replaceAll("\\b've\\b"," have")
    toClean = toClean.replaceAll("\\bi'd\\b|\\bid\\b|\\bi d\\b","i would")
    toClean = toClean.replaceAll("\\bhe'd\\b|\\bhed\\b|\\bhe d\\b","he would")
    toClean = toClean.replaceAll("\\bshe'd\\b|\\bshed\\b|\\bshe d\\b","she would")
    toClean = toClean.replaceAll("\\bwe'd\\b|\\bwed|\\bwe d\\b","we would")
    toClean = toClean.replaceAll("\\bthey'd\\b|\\btheyd\\b|\\bthey d\\b","they would")
    toClean = toClean.replaceAll("\\bit'd\\b|\\bitd\\b|\\bit d\\b","you would")
    toClean = toClean.replaceAll("\\b'll\\b"," will")
    toClean = toClean.replaceAll("\\bwon't\\b|\\bwont\\b","will not")
    toClean = toClean.replaceAll("\\b's\\b"," is")
    toClean = toClean.replaceAll("\\b'd\\b"," would")
    toClean = toClean.replaceAll("\\b're\\b"," are")
    toClean = toClean.replaceAll("\\bi'm\\b|\\bim\\b","i am")
    toClean = toClean.replaceAll("\\blet's\\b","let us")
    toClean = toClean.replaceAll("\\bma'am\\b","madam")

    toClean
  }

  def cleanToxicWords(c : String) : String={
    var toClean = c
    toClean = toClean.replaceAll("(f)(u|[^a-z0-9 ])(c|[^a-z0-9 ])(k|[^a-z0-9 ])([^ ])*|fuc*$|fck|\\bf\\b|\\bf uck\\b|\\bf ck\\b|\\bf+u+\\b","fuck")
    toClean = toClean.replaceAll("a s s|arse","ass")
    toClean = toClean.replaceAll("b!itch|biatch|b i t c h|bi[t]+ch","bitch")
    toClean = toClean.replaceAll("ba[s|z]+t[e|a]+rd","bastard")
    toClean = toClean.replaceAll("d i c k","dick")
    toClean = toClean.replaceAll("c u n t","cunt")
    toClean = toClean.replaceAll("ni[g]+a|nigr|n3gr|n i g g e r","nigger")
    toClean = toClean.replaceAll("stfu","shit the fuck up")
    toClean = toClean.replaceAll("pussi|pu[s]+y","pussy")
    toClean = toClean.replaceAll("fag|f a g g o t|faggit|fagot","faggot")
    toClean = toClean.replaceAll("moth f|motherucker","motherfucker")
    toClean = toClean.replaceAll("w h o r e","whore")
    toClean = toClean.replaceAll("sh\\*tty","shit")

    toClean
  }

  def cleanPunctuation(c: String) : String = {
    var toClean = c
    toClean = toClean.replaceAll("\\.|\\?|!|,|;|'|\"","")

    toClean
  }



  def cleanCommentOperation(dataFrame : DataFrame) : DataFrame ={
    val cleanComment = udf((col: String) =>
    {
      var comment = col.toString
      comment = comment.toLowerCase()
      comment = comment.trim()
      comment = cleanRegularWords(comment)
      comment = cleanToxicWords(comment)
      comment = cleanPunctuation(comment)

      comment
    })

    dataFrame.withColumn("comment_text",cleanComment(dataFrame("comment_text")))
  }

  def prepareDF(dataFrame: DataFrame) : DataFrame = {
    var newDF = dataFrame
    newDF = cleanNullValues(newDF, Array("target","comment_text"))
    newDF = cleanTarget(newDF)
    newDF = cleanCommentOperation(newDF)

    newDF
  }

  def prepareDfForPrediction(dataFrame: DataFrame) : DataFrame = {
    var newDF = dataFrame
    newDF = cleanNullValues(newDF,Array("comment_text"))
    newDF = cleanCommentOperation(newDF)


    newDF
  }


}
