#include <owl/owl.h>

#include "viewer.h"

#define LOG(message)                                            \
  std::cout << OWL_TERMINAL_BLUE;                               \
  std::cout << "#owl.sample(main): " << message << std::endl;   \
  std::cout << OWL_TERMINAL_DEFAULT;
#define LOG_OK(message)                                         \
  std::cout << OWL_TERMINAL_LIGHT_BLUE;                         \
  std::cout << "#owl.sample(main): " << message << std::endl;   \
  std::cout << OWL_TERMINAL_DEFAULT;

int main(int ac, char **av)
{
    Scene rt_scene;
    //rt_scene.load_obj_into_scene("data/stanford_bunny.obj");
    //rt_scene.load_obj_into_scene("D:\\Bureau\\Repos\\M1\\m-1-synthese\\tp2\\data\\xyzrgb_dragon.obj");
    //rt_scene.load_obj_into_scene("D:\\Bureau\\Repos\\M1\\m-1-synthese\\tp2\\data\\burger_tom300.obj");
    rt_scene.load_obj_into_scene("../../common_data/grass_medium_01_8k.obj");

    rt_scene.load_skysphere("../../common_data/AllSkyFree_Sky_EpicGloriousPink_Equirect.png");
    //rt_scene.load_skysphere("D:/Bureau/AllSkyFree_Sky_ClearBlueSky_Equirect.png");

    Viewer viewer(rt_scene);
    viewer.camera.setOrientation(Viewer::INIT_CAMERA_POSITION, Viewer::INIT_CAMERA_LOOKAT, Viewer::INIT_CAMERA_UP, owl::viewer::toDegrees(acosf(Viewer::INIT_CAMERA_FOVY)));
    viewer.enableInspectMode();

    viewer.showAndRun();

    return 0;
}
