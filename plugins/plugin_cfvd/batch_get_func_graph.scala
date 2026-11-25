import java.io.File
import scala.io.Source

@main def exec(listFile: String) = {
  val lines = Source.fromFile(listFile).getLines().filter(_.trim.nonEmpty).toList
  for (filename <- lines) {
    val edgesF = new File(filename + ".edges.json")
    val nodesF = new File(filename + ".nodes.json")
    if (!edgesF.exists() || !nodesF.exists()) {
      try {
        importCode.c(filename)
        run.ossdataflow
        cpg.graph.E.map(node=>List(node.inNode.id, node.outNode.id, node.label, node.propertiesMap.get("VARIABLE"))).toJson |> filename + ".edges.json"
        cpg.graph.V.map(node=>node).toJson |> filename + ".nodes.json"
        delete
      } catch {
        case e: Throwable =>
          println(s"[JoernBatch] Error processing $filename: ${e.getMessage}")
          try { delete } catch { case _: Throwable => }
      }
    }
    println(s"[PROCESSED] $filename")
    System.out.flush()
  }
}
