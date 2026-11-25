import java.io.{BufferedWriter, FileWriter, File}

@main def exec(
    cpgPath:   String,
    outputDir: String,
    minNodes:  Int = 1,
    maxNodes:  Int = 499,
    minEdges:  Int = 1
) = {

  println(s"[joern_to_vul_lmgnn] Starting. CPG: $cpgPath  Output: $outputDir")
  println(s"[joern_to_vul_lmgnn] Filters - minNodes=$minNodes  maxNodes=$maxNodes  minEdges=$minEdges")

  new File(outputDir).mkdirs()
  scala.util.Try(delete("livdb_cpg_proj"))
  val projDir = new File("workspace/livdb_cpg_proj")
  if (projDir.exists()) {
    def delRec(f: File): Unit = {
      if (f.isDirectory) { val k = f.listFiles(); if (k != null) k.foreach(delRec) }
      f.delete()
    }
    scala.util.Try(delRec(projDir))
  }
  importCpg(cpgPath, "livdb_cpg_proj")

  def escapeJson(s: String): String =
    if (s == null) ""
    else s.replace("\\", "\\\\")
          .replace("\"", "\\\"")
          .replace("\n", "\\n")
          .replace("\r", "\\r")
          .replace("\t", "\\t")
          .replace("\b", "\\b")
          .replace("\f", "\\f")

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

        val nodeMap = scala.collection.mutable.LinkedHashMap[Long, (String, String, Int, Int, String, String)]()
        allAstNodes.foreach { n =>
          val code = scala.util.Try(n.code).getOrElse("")
          val ntype = n.label
          val line = scala.util.Try(n.lineNumber.get.toInt).getOrElse(-1)
          val col  = scala.util.Try(n.columnNumber.get.toInt).getOrElse(-1)
          val mfn  = scala.util.Try(n.property("METHOD_FULL_NAME").toString).getOrElse("")
          val tfn  = scala.util.Try(n.property("TYPE_FULL_NAME").toString).getOrElse("")
          nodeMap(n.id) = (code, ntype, line, col, mfn, tfn)
        }

        val domainIds = nodeMap.keySet

        val edgeBuf = scala.collection.mutable.ArrayBuffer[(Long, String, Long)]()
        allAstNodes.foreach { src =>
          if (domainIds.contains(src.id)) {
            src.outE.foreach { edge =>
              scala.util.Try {
                val dstId = edge.inNode.id
                if (domainIds.contains(dstId)) {
                  edgeBuf += ((src.id, edge.label, dstId))
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

          val nodeSb = new java.lang.StringBuilder(nodeMap.size * 128)
          idList.zipWithIndex.foreach { case (id, i) =>
            if (i > 0) nodeSb.append(',')
            val (code, ntype, line, col, mfn, tfn) = nodeMap(id)
            nodeSb.append(s"""{"id":$i,"raw_id":$id,"code":"${escapeJson(code)}","node_type":"${escapeJson(ntype)}","line":$line,"col":$col,"mfn":"${escapeJson(mfn)}","tfn":"${escapeJson(tfn)}"}""")
          }

          val edgeSb = new java.lang.StringBuilder(edgeBuf.size * 32)
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

          val json =
            s"""{"func_name":"${escapeJson(methodName)}","label":$label,"nodes":[${nodeSb}],"edges":[${edgeSb}]}"""

          val bw = new BufferedWriter(new FileWriter(outPath), 65536)
          try { bw.write(json) } finally { bw.close() }

          written += 1
          if (written % 2000 == 0) {
            println(s"[joern_to_vul_lmgnn] Progress: exported $written functions to JSON...")
          }
        }
      }
    }

  println(s"[joern_to_vul_lmgnn] Done. Written=$written  Skipped(filters)=$skipped  Output: $outputDir")
}
