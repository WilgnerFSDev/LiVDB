import java.io.{BufferedWriter, FileWriter, File}

@main def exec(
    cpgPath:   String,
    outputDir: String,
    minNodes:  Int = 1,
    maxNodes:  Int = 500,
    minEdges:  Int = 1
) = {

  println(s"[joern_reveal] Starting extraction. CPG: $cpgPath  Output: $outputDir")
  println(s"[joern_reveal] Filters - minNodes=$minNodes  maxNodes=$maxNodes (ReVeal <= 500)  minEdges=$minEdges")

  new File(outputDir).mkdirs()
  importCpg(cpgPath)

  def escapeJson(s: String): String = {
    if (s == null) ""
    else s.replace("\\", "\\\\")
          .replace("\"", "\\\"")
          .replace("\n", "\\n")
          .replace("\r", "\\r")
          .replace("\t", "\\t")
          .replace("\b", "\\b")
          .replace("\f", "\\f")
  }

  def labelFromFilename(filename: String): Int = {
    val base  = new File(filename).getName.replaceAll("\\.(c|cpp|h|hpp)$", "")
    val parts = base.split("_")
    scala.util.Try(parts.last.toInt).getOrElse(0)
  }

  var written = 0
  var skipped = 0

  cpg.method
    .filter(m =>
        m.filename != "<empty>" &&
        (m.filename.endsWith(".c") || m.filename.endsWith(".cpp") || m.filename.endsWith(".h") || m.filename.endsWith(".hpp")) &&
        !m.name.startsWith("<")
    )
    .foreach { method =>

      val allAstNodes = method.ast.l
      val astCount    = allAstNodes.size

      if (astCount < minNodes || astCount > maxNodes) {
        skipped += 1
      } else {

        val methodName = method.name
        val filename   = method.filename
        val label      = labelFromFilename(filename)

        val nodeMap = scala.collection.mutable.LinkedHashMap[Long, (String, String)]()
        allAstNodes.foreach { n =>
          val code     = scala.util.Try(n.code).getOrElse("")
          val nodeType = n.label
          nodeMap(n.id) = (code, nodeType)
        }

        val domainIds = nodeMap.keySet
        val edgeBuf   = scala.collection.mutable.ArrayBuffer[(Long, String, Long)]()

        allAstNodes.foreach { src =>
          src.astChildren.foreach { dst =>
            if (domainIds.contains(dst.id) && src.id != dst.id) {
              edgeBuf += ((src.id, "AST", dst.id))
            }
          }
        }

        allAstNodes.foreach { src =>
          if (domainIds.contains(src.id)) {
            src.outE.foreach { edge =>
              scala.util.Try {
                val dstId = edge.dst.id
                if (domainIds.contains(dstId) && src.id != dstId) {
                  val edgeLabel = edge.label

                  if (edgeLabel != "AST") {
                    edgeBuf += ((src.id, edgeLabel, dstId))
                  }
                }
              }
            }
          }
        }

        val numEdges = edgeBuf.size

        if (numEdges < minEdges) {
          skipped += 1
        } else {

          val idList    = nodeMap.keys.toSeq
          val idToIndex = idList.zipWithIndex.toMap

          val nodeSb = new java.lang.StringBuilder(nodeMap.size * 96)
          idList.zipWithIndex.foreach { case (id, i) =>
            if (i > 0) nodeSb.append(',')
            val (code, ntype) = nodeMap(id)
            nodeSb.append(s"""{"id":$i,"code":"${escapeJson(code)}","node_type":"${escapeJson(ntype)}"}""")
          }

          val edgeSb = new java.lang.StringBuilder(edgeBuf.size * 36)
          var firstEdge = true
          edgeBuf.foreach { case (src, etype, dst) =>
            if (idToIndex.contains(src) && idToIndex.contains(dst)) {
              if (!firstEdge) edgeSb.append(',')
              edgeSb.append(s"""[${idToIndex(src)},"${escapeJson(etype)}",${idToIndex(dst)}]""")
              firstEdge = false
            }
          }

          val safeMethod = methodName.replaceAll("[^a-zA-Z0-9_\\-]", "_")
          val safeFile   = new File(filename).getName.replaceAll("\\.(c|cpp|h|hpp)$", "")
          val outPath    = s"$outputDir/${safeFile}__${safeMethod}.json"

          val json = s"""{"func_name":"${escapeJson(methodName)}","label":$label,"nodes":[${nodeSb}],"edges":[${edgeSb}]}"""

          val bw = new BufferedWriter(new FileWriter(outPath), 65536)
          try { bw.write(json) } finally { bw.close() }

          written += 1
        }
      }
    }

  println(s"[joern_reveal] Extraction complete. Written=$written  Skipped(filters)=$skipped  Output: $outputDir")
}
