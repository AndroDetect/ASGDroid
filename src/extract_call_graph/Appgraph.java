import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import soot.Scene;
import soot.jimple.infoflow.InfoflowConfiguration;
import soot.jimple.infoflow.android.InfoflowAndroidConfiguration;
import soot.jimple.infoflow.android.SetupApplication;
import soot.jimple.toolkits.callgraph.CallGraph;


public class Appgraph {
	public static void main(String[] args) {
		String appToRun = args[0];
		String androidPlatform = args[1];
		String filename = appToRun + ".txt";
		InfoflowAndroidConfiguration config = new InfoflowAndroidConfiguration();
		config.getAnalysisFileConfig().setAndroidPlatformDir(androidPlatform);
		config.getAnalysisFileConfig().setTargetAPKFile(appToRun);
		config.setMergeDexFiles(true);
		config.setCodeEliminationMode(InfoflowConfiguration.CodeEliminationMode.NoCodeElimination);
		config.setEnableReflection(true);
		config.setCallgraphAlgorithm(InfoflowConfiguration.CallgraphAlgorithm.CHA);
		config.getAccessPathConfiguration().setAccessPathLength(1);

		SetupApplication app = new SetupApplication(config);

		app.constructCallgraph();
		
		CallGraph cg = Scene.v().getCallGraph();
		try (BufferedWriter writer = new BufferedWriter(
				new FileWriter(
						new File(filename)))
		){
			writer.write(Scene.v().getCallGraph().toString());
		}
		catch (IOException e){
			System.out.println("An error occurred");
		}
	}
}
